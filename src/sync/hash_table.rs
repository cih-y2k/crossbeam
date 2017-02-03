extern crate fnv;
use std::sync::atomic::Ordering::{self, Acquire, Release, Relaxed, AcqRel, SeqCst};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, fence};
use std::hash::{Hash, Hasher};
// use std::collections::hash_map::DefaultHasher;
use std::mem;
use std::fmt::Debug;
use std::string::String;
use std::io::Write;
use std::io::stdout;
use std::isize;

use mem::epoch::{self, Atomic, Owned, Guard, Shared};

const SEG_SHIFT: usize = 3;
const SEG_SIZE: usize = 1 << SEG_SHIFT;

// The threshold after which a segment is considered to be too full. When a bucket has
// `last_snapshot_len` greater than this value, it votes for a grow operation.
const SEG_LOAD_FACTOR: usize = 8;

// The cadence by which calls to `add_opt` attempt to clean up buckets and hence update their
// snapshot lengths (thereby facilitating a vote for growth).
const SEG_CLEANUP_EVERY: usize = 16;
// remove operations always amount to appending a single record
const ALWAYS_APPEND_REM: bool = false;

// make an effort to clean up garbage when marking a HashTable for deletion
const CAMPFIRE_RULES: bool = false;
const DBG: bool = false;

/// A simple wrapper around a `Vec` that provide simple set operations. Used to track the live
/// elements in a bucket during growing and cleaning operations.
struct VecSet<T>(Vec<T>);

impl<T: PartialEq<T>> VecSet<T> {
    #[inline]
    /// O(1) insert
    fn insert(&mut self, t: T) {
        self.0.push(t);
    }

    #[inline]
    /// O(n) contains, where n is number of times insert has been called
    fn contains(&mut self, t: &T) -> bool {
        self.0.contains(t)
    }

    #[inline]
    fn new() -> VecSet<T> {
        VecSet(Vec::new())
    }
}

#[derive(Debug)]
/// `LookupResult` is like `Option`, but it disambiguates between values that were once in a
/// structure but have since been deleted. This is used in the `LazySet` lookup method; note that
/// in that case `LazySet` may still return `NotFound` for values that have since been deleted, but
/// whose records have been removed in a `cleanup` operation. Public callers into `LazySet` should
/// therefore only take this on an advisory basis unless all `remove` and `add` calls have had the
/// `growing` parameter passed as `true` (in which case, no cleanups will have been called, but
/// there will be an ensuing space leak).
///
/// This functionality is used by `HashTable` to help with the case of a key being removed from the
/// table before its actual record has been copied into the new bucket array. In this case, lookups
/// in the current bucket array will return `Deleted` and the overall lookup can return `None`.
pub enum LookupResult<T> {
    Found(T),
    Deleted,
    NotFound,
}

impl<T> LookupResult<T> {
    #[inline]
    pub fn is_found(&self) -> bool {
        if let LookupResult::Found(_) = *self {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_deleted(&self) -> bool {
        if let LookupResult::Deleted = *self {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_notfound(&self) -> bool {
        if let LookupResult::NotFound = *self {
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct Stamp(AtomicUsize);

impl Stamp {
    #[inline]
    pub fn init(&self, v: usize) {
        self.0.store(v & ((!0) << 1), Relaxed);
    }

    #[inline]
    pub fn delete(&self, o1: Ordering, o2: Ordering) {
        self.0.store(self.0.load(o1) | 1, o2)
    }

    #[inline]
    pub fn is_deleted(&self, o: Ordering) -> bool {
        (self.0.load(o) & 1) == 1
    }

    #[inline]
    pub fn matches(&self, h: usize, o: Ordering) -> bool {
        (self.0.load(o) | 1) == (h | 1)
    }
}


/// a HashHandle owns both a direct pointer to a HashTable and an Arc for a HashHandleInternal. This
/// is so we can get ref-counting from Arc, but still only have one pointer-dereference for those
/// interacting with the underlying hash table.
struct HashHandleInternal<K: Eq + Hash + Clone, V: Clone> {
    ptr: Atomic<HashTable<K, V>>,
}

impl<K: Eq + Hash + Clone, V: Clone> Drop for HashHandleInternal<K, V> {
    fn drop(&mut self) {
        unsafe {
            cleanup(&mut self.ptr);
        }
    }
}

/// HashHandle provides a safe interface for interacting with a HashTable. It behaves like an Arc,
/// but when it goes out of scope it calls a safe memory reclamation method for a HashTable. This
/// struct provides a number of wrapper methods for the underlying hash table.
pub struct HashHandle<K: Eq + Hash + Clone, V: Clone> {
    table: Atomic<HashTable<K, V>>,
    rc: Arc<HashHandleInternal<K, V>>,
    // used for caching segment allocations if the CAS in `search_forward` fails.
    spare_allocs: Vec<Owned<Segment<(K, Option<V>)>>>,
}


impl<K: Eq + Hash + Clone, V: Clone> Clone for HashHandle<K, V> {
    fn clone(&self) -> HashHandle<K, V> {
        let new_handle = HashHandle {
            table: Atomic::null(),
            rc: self.rc.clone(),
            spare_allocs: Vec::new(),
        };
        new_handle.table.store_shared(self.table.load(Relaxed, &epoch::pin()), Relaxed);
        new_handle
    }
}

#[inline]
fn is_power_of_two(u: usize) -> bool {
    u & (!u + 1) == u
}

// basic wrapper methods for accessing the table within a HashHandle.
impl<K: Eq + Hash + Clone, V: Clone> HashHandle<K, V> {
    pub fn new() -> HashHandle<K, V> {
        HashHandle::with_capacity_grow(8, 1024)
    }

    pub fn with_capacity_grow(growth_rate: usize, start_size: usize) -> HashHandle<K, V> {
        if !(is_power_of_two(growth_rate) && is_power_of_two(start_size)) {
            panic!("passed invalid sizes for growth rate ({}) and start size ({})",
                   growth_rate,
                   start_size)
        }
        let new_table = Owned::new(HashTable::new_val(growth_rate, start_size));
        let new_atomic = Atomic::null();
        new_atomic.store(Some(new_table), Relaxed);
        let snd_atomic = Atomic::null();
        snd_atomic.store_shared(new_atomic.load(Relaxed, &epoch::pin()), Relaxed);
        let new_handle = HashHandle {
            table: new_atomic,
            rc: Arc::new(HashHandleInternal { ptr: snd_atomic }),
            spare_allocs: Vec::new(),
        };
        new_handle
    }

    /// Call arbitrary `HashTable` operations on the underlying table of the HashHandle. This is not
    /// recommended as `HashHandle` keeps a vector of spare segments to avoid excess allocations
    /// when growing a hash bucket. This is currently here for two reasons:
    ///     * It allows for `insert` and `remove` operations that do not borrow the HashHandle as
    ///     mutable.
    ///     * It allows for lookup operations, the lookup API has yet to stabilize for `HashHandle`
    pub fn with_table<R, F: FnMut(&HashTable<K, V>) -> R>(&self, mut f: F) -> R {
        let guard = epoch::pin();
        f(*self.table.load(Relaxed, &guard).unwrap())
    }

    pub fn insert(&mut self, k: K, v: V) {
        let guard = epoch::pin();
        self.table.load(Relaxed, &guard).unwrap().insert(k, v, &mut self.spare_allocs);
    }

    pub fn remove(&mut self, key: &K) -> bool {
        let guard = epoch::pin();
        self.table.load(Relaxed, &guard).unwrap().remove(key, &mut self.spare_allocs)
    }

    pub fn contains(&self, key: &K) -> bool {
        let guard = epoch::pin();
        self.table.load(Relaxed, &guard).unwrap().contains(key)
    }
}

pub struct HashTable<K: Eq + Hash + Clone, V: Clone> {
    buckets: Atomic<Vec<LazySet<K, V>>>,
    prev_buckets: Atomic<Vec<LazySet<K, V>>>,
    grow_factor: usize,
    grow_votes: AtomicIsize,
    growing: AtomicBool,
}

#[allow(unused_mut)]
/// Unsafely reclaim all memory associated with a hash table. This is effectively a safe `Drop`
/// method for HashTable that uses the epoch subsystem to enqueue callbacks. It is unsafe because
/// it should only be called when there are no more possible reference for the hash table. This is
/// called safely when a HashHandle has no remaining references.
///
/// TODO: if cleanup is called and then the program shuts down, it is possible for some of these
/// callbacks to not get run. This is _fine_ from a memory safety perspective, but it is bad if the
/// caller expects `Drop` to be called on all of the elements that they add to the structure. The
/// mitigation for this would be to add some sort of finalizer for global metadata in the epoch
/// subsystem.
/// Having local epoch-gc would also fix this problem as we could simply run all destructors and
/// free anything in `Drop`.
///
/// TODO: This should probably just be private.
pub unsafe fn cleanup<K: Eq + Hash + Clone, V: Clone>(ptr: &mut Atomic<HashTable<K, V>>) {
    // Note that we assume no one is in between the cas of the `growing` bool and the re-assignment
    // of `prev_buckets` and `buckets` pointers. This happens *before* any deferred closures are
    // enqueued by `try_grow`, so it will be satisfied if no threads other than the current one
    // maintain a reference to the table.
    //
    // Because the above condition is satisfied, we can free the current buckets (but not prev, if
    // it exists), the HashTable struct itslef, and enqueue the destructors for individual
    // elements. This is all done three epochs in the future becuase `try_grow` enqueues
    // destructors two epochs into the future that modify `ptr`'s metadata; those may not have run
    // yet.
    {
        let raw_ptr_self = ptr.ptr.load(Relaxed) as usize;
        let guard = epoch::pin();
        let buckets_ptr: *mut Vec<LazySet<K, V>> =
            ptr.load(Relaxed, &guard).unwrap().buckets.load(Relaxed, &guard).unwrap().as_raw();
        let buckets_raw = buckets_ptr as usize;
        guard.destructor(Box::new(move || {
            let guard = epoch::pin();
            guard.destructor(Box::new(move || {
                let guard = epoch::pin();
                guard.destructor(Box::new(move || {
                    let guard = epoch::pin();
                    let mut buckets_ptr =
                        (buckets_raw as *mut Vec<LazySet<K, V>>).as_mut().unwrap();
                    while let Some(s) = buckets_ptr.pop() {
                        delete_set(&s, &guard);
                    }
                    guard.unlinked(Shared::from_ref(buckets_ptr));
                    guard.unlinked(Shared::from_raw(raw_ptr_self as *mut HashTable<K, V>).unwrap());
                    guard.migrate_garbage();
                }));
                guard.migrate_garbage();
            }));
        }));
        guard.migrate_garbage();
    }
    // Collect all epochs' garbage if there aren't any active threads using the epoch scheme.
    if CAMPFIRE_RULES {
        for _i in 0..4 {
            epoch::pin().quiesced_collect();
        }
        println!("done");
    }
}

/// iterate over a LazySet and enqueue delete and destructor operations for all of its
/// segments and nodes. This is unsafe for the same reason that guard.unliked is unsafe: it tacitly
/// assumes that no one will acquire a new reference to any of `bucket`'s contents.
unsafe fn delete_set<'a, K: Eq + Hash + Clone, V>(bucket: &LazySet<K, V>, guard: &'a Guard) {
    if DBG {
        let fails = bucket.cas_failures.load(Relaxed);
        if fails > 0 {
            println!("{} cas failures", fails);
        }
    }
    // consider batching these frees into a single callback
    let mut cur_seg = bucket.head.load(Relaxed, guard);
    while let Some(s) = cur_seg {
        for i in 0..SEG_SIZE {
            if let Some(data) = s.data.get_unchecked(i).data.load(Relaxed, guard) {
                destroy_and_free(data, guard);
            }
        }
        guard.unlinked(s);
        cur_seg = s.next.load(Relaxed, guard);
    }
}

impl<K: Eq + Hash + Clone, V: Clone> HashTable<K, V> {
    /// creates and initializes a new array of buckets.
    fn make_buckets(num_buckets: usize) -> Owned<Vec<LazySet<K, V>>> {
        let res: Owned<Vec<LazySet<K, V>>> =
            Owned::new((0..num_buckets).map(|_| LazySet::new()).collect());
        assert_eq!(res.len(), num_buckets);
        res
    }

    /// Returns an Atomic<HashTable>, the caller is responsible for manually calling cleanup on
    /// this value (which is why this is marked as unsafe). It is recommended to instead interact
    /// with a hash table through a `HashHandle`, as that is safe and provides better performance
    /// due to the cached allocations.
    pub unsafe fn new_atomic(grow_factor: usize, start_size: usize) -> Atomic<HashTable<K, V>> {
        // both assumed to be powers of two
        if !(is_power_of_two(grow_factor) && is_power_of_two(start_size)) {
            panic!("passed invalid sizes for growth rate ({}) and start size ({})",
                   grow_factor,
                   start_size)
        }
        let res = HashTable {
            buckets: Atomic::null(),
            prev_buckets: Atomic::null(),
            grow_factor: grow_factor,
            grow_votes: AtomicIsize::new(0),
            growing: AtomicBool::new(false),
        };
        res.buckets.store(Some(HashTable::make_buckets(start_size)), Relaxed);
        let r_ptr = Atomic::null();
        let _ = r_ptr.store(Some(Owned::new(res)), Relaxed);
        r_ptr
    }

    fn new_val(grow_factor: usize, start_size: usize) -> HashTable<K, V> {
        // both assumed to be powers of two
        let res = HashTable {
            buckets: Atomic::null(),
            prev_buckets: Atomic::null(),
            grow_factor: grow_factor,
            grow_votes: AtomicIsize::new(0),
            growing: AtomicBool::new(false),
        };
        res.buckets.store(Some(HashTable::make_buckets(start_size)), Relaxed);
        res
    }

    /// `try_grow` contains the algorithm for changing the hash table size so
    /// trans(self.buckets.len()). We assume that this new value is a power of two.
    /// This algorithm is complicated and enqueues multiple layers of callbacks in the epoch
    /// subsystem.
    fn try_grow<'a, F: Fn(usize) -> usize>(&self, guard: &'a Guard, trans: F) {
        // called once bucket has quiesced; this operation copies and re-hashes all of the elements
        // in 'bucket', removing duplicate entries along the way. It then registers reclaiming
        // callbacks to be performed for bucket's members, thereby assuming that the function's
        // results will be backfilled into a newer array of buckets.
        //
        // The return value is a map from hash -> Vec<(stamp, key, value)>
        unsafe fn rebucket<'a, K: Hash + Eq + Clone, V: Clone>
            (bucket: &LazySet<K, V>,
             guard: &'a Guard,
             new_mod: usize)
             -> Vec<(usize, Vec<(usize, K, V)>)> {
            let mut res: Vec<(usize, Vec<(usize, K, V)>)> = Vec::new();
            for (ref _hash, tup) in bucket.iter_live(Relaxed, guard) {
                let mut hasher = fnv::FnvHasher::default();
                tup.0.hash(&mut hasher);
                let hash = hasher.finish() as usize;
                let bucket_ind = hash & (new_mod - 1);
                let mut to_insert = Some((hash, tup.0.clone(), tup.1.clone().unwrap()));
                for &mut (bkt, ref mut vec) in res.iter_mut() {
                    if bkt == bucket_ind {
                        vec.push(to_insert.take().unwrap());
                        break;
                    }
                }
                if let Some(t) = to_insert {
                    res.push((bucket_ind, vec![t]));
                }
            }
            let raw_ptr = bucket as *const LazySet<K, V> as *mut LazySet<K, V> as usize;
            guard.destructor(Box::new(move || {
                let guard = epoch::pin();
                let raw_ptr = raw_ptr as *mut LazySet<K, V>;
                delete_set(raw_ptr.as_ref().expect("rebucket/destruct_raw_ptr"), &guard);
            }));
            res
        }
        // Figure out if there is an ongoing grow operation. We are not allowed to do these
        // concurrently.
        if self.growing.compare_and_swap(false, true, Relaxed) {
            return;
        }
        let start_votes = self.grow_votes.load(Relaxed);
        let cur_buckets = self.buckets.load(Relaxed, guard).unwrap();
        let new_mod = trans(cur_buckets.len());
        let new_buckets = HashTable::make_buckets(new_mod);
        // copy cur_buckets to prev_buckets, cas new_buckets into cur_buckets. Now all newly issued
        // mutation operations will operate _only_ on buckets. It is important that these values
        // happen after one another and that they both happen after growing is set to true, as the
        // growing flag is what prevents new delete operations from mutating an old bucket.
        //
        // Note: After this, we will have to contend with mutation operations that are in progress
        // on prev_buckets.
        assert!(self.prev_buckets.cas_shared(None, Some(cur_buckets), SeqCst));
        assert!(self.buckets.cas(Some(cur_buckets), Some(new_buckets), SeqCst).is_ok());

        // Now we need to update metadata at some point in the future, this stuff is all just hacks
        // around the borrow checker. I am very sorry about this.
        let raw_old = self.prev_buckets.load(Relaxed, guard).unwrap().as_raw() as usize;
        let raw_new = self.buckets.load(Relaxed, guard).unwrap().as_raw() as usize;
        let self_growing = (&self.growing) as *const AtomicBool as *mut AtomicBool as usize;
        let self_prev = (&self.prev_buckets) as
            *const Atomic<Vec<LazySet<K, V>>> as
            *mut Atomic<Vec<LazySet<K, V>>> as usize;
        let self_grow_votes = (&self.grow_votes) as *const AtomicIsize as *mut AtomicIsize as usize;
        unsafe {
            guard.destructor(Box::new(move || {
                // Once all in-progress operations complete, we know that there will be no
                // additional mutation operations, so we can begin to migrate buckets. However we
                // still can't free memory because it is only after all of the buckets have been
                // migrated that we can expect new 'lookup' operations to skip old_buckets. Thus,
                // the rebucket operation appends another set of callbacks to be executed after
                // _this_ epoch has quiesced.
                let raw_old2 = raw_old; // copy raw_old as it is moved into second destructor
                let old_ptr = (raw_old as *mut Vec<LazySet<K, V>>).as_ref().expect("raw_old1");
                let new_ptr = (raw_new as *mut Vec<LazySet<K, V>>).as_ref().expect("raw_new");
                let guard = epoch::pin();
                for i in old_ptr.iter() {
                    let mut new_buckets = rebucket(i, &guard, new_mod);
                    while let Some((hash, vals)) = new_buckets.pop() {
                        new_ptr[hash].back_fill_stamped(vals);
                    }
                }
                guard.destructor(Box::new(move || {
                    // Now we are sure there are no more concurrent readers of the old bucket
                    // array, so it is safe to reclaim it and mark growing as complete, as well as
                    // remove the grow_votes that we acted on earlier.
                    let prev_ptr = (self_prev as *mut Atomic<Vec<LazySet<K, V>>>)
                        .as_ref()
                        .expect("prev_ptr");
                    prev_ptr.store(None, Relaxed);
                    // assert!(prev_ptr.cas_shared(Some(Shared::from_ref(old_ptr)), None, AcqRel));
                    // free the vector
                    let grow_votes_ptr = (self_grow_votes as *mut AtomicIsize)
                        .as_ref()
                        .expect("self_grow_votes");
                    grow_votes_ptr.fetch_add(-start_votes, Relaxed);
                    let growing_ptr = (self_growing as *mut AtomicBool)
                        .as_ref()
                        .expect("self_growing");
                    assert!(growing_ptr.compare_and_swap(true, false, AcqRel));
                    fence(AcqRel);
                    mem::drop(Box::from_raw(raw_old2 as *mut Vec<LazySet<K, V>>));
                }));
            }));
        }
    }

    fn get_bucket<'a>(buckets: Shared<'a, Vec<LazySet<K, V>>>,
                      key: &K)
                      -> (usize, &'a LazySet<K, V>) {
        // TODO return an option for the bucket in the old array
        let mut hasher = fnv::FnvHasher::default();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let vec = buckets.as_slice();
        // unsafe { vec.get_unchecked((hash as usize) % vec.len()) }
        (hash, unsafe { vec.get_unchecked(hash & (vec.len() - 1)) })
    }

    #[inline]
    fn grow_threshold(&self, guard: &Guard) -> usize {
        let len = self.buckets.load(Relaxed, guard).unwrap().len();
        len - (len >> 3)
    }

    fn insert(&self, key: K, val: V, vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>) {
        let guard = epoch::pin();
        let (mut hash, mut bucket) =
            HashTable::get_bucket(self.buckets.load(Relaxed, &guard).unwrap(), &key);
        if self.grow_factor != 1 && bucket.last_snapshot_len.load(Relaxed) > SEG_LOAD_FACTOR {
            if self.grow_votes.fetch_add(1, Relaxed) as usize > self.grow_threshold(&guard) {
                self.try_grow(&guard, |n| n * self.grow_factor);
                let (new_hash, new_bucket) =
                    HashTable::get_bucket(self.buckets.load(Relaxed, &guard).unwrap(), &key);
                hash = new_hash;
                bucket = new_bucket;
            }
        }
        bucket.add_stamped(hash, key, val, self.growing.load(Acquire), vec)
    }

    pub fn lookup<'b, 'a: 'b>(&self, key: &'a K, guard: &'b Guard) -> Option<&'b V> {
        let (hash, bucket) = HashTable::get_bucket(self.buckets.load(Relaxed, &guard).unwrap(),
                                                   key);
        // This is somewhat subtle, see the comment above the definition of LookupResult.
        match bucket.lookup_stamped(hash, key, guard) {
            LookupResult::Found(n) => Some(n),
            LookupResult::Deleted => None,
            LookupResult::NotFound => {
                if let Some(buckets) = self.prev_buckets.load(Relaxed, &guard) {
                    let (hash, bucket) = HashTable::get_bucket(buckets, key);
                    if let LookupResult::Found(elt) = bucket.lookup_stamped(hash, key, guard) {
                        return Some(elt);
                    }
                }
                None
            }
        }
    }

    pub fn scoped_lookup<F, R>(&self, key: &K, mut f: F) -> R
        where F: FnMut(Option<&V>) -> R
    {
        let guard = epoch::pin();
        f(self.lookup(key, &guard))
    }

    pub fn contains(&self, key: &K) -> bool {
        self.scoped_lookup(key, |o| o.is_some())
    }

    fn remove(&self, key: &K, vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>) -> bool {
        let guard = epoch::pin();
        let (hash, bucket) = HashTable::get_bucket(self.buckets.load(Relaxed, &guard).unwrap(),
                                                   key);
        bucket.remove_stamped(hash,
                              key,
                              if ALWAYS_APPEND_REM {
                                  true
                              } else {
                                  self.growing.load(Relaxed)
                              },
                              vec)
    }
}

#[derive(Debug)]
/// A `LazySet` is our choice of hash bucket, but it can serve as a wait-free set data-structure
/// (with lock-free back_fill operation, though it is unclear what the practical use of that is).
/// It is lazy because removal operations only logically delete elements by flipping a bit, and
/// there is no guarantee that the elements' memory will ever be reclaimed if there are no
/// additional operations performed on the set before the entire `LazySet` is deleted.
///
/// To reclaim the memory associated with a `LazySet` that will not acquire any new references,
/// call the unsafe `delete_set` operation.
pub struct LazySet<K: Eq + Hash + Clone, V> {
    head_ptr: AtomicIsize,
    head: Atomic<Segment<(K, Option<V>)>>,
    delete_ptr: AtomicUsize,
    cleaning: AtomicBool, // only one thread should run cleanup at a time
    die: AtomicBool, // for debugging purposes
    cas_failures: AtomicUsize, // debugging
    last_snapshot_len: AtomicUsize,
}

/// destroy_and_free usnafely takes ownership and frees a shared pointer when the epoch-based
/// reclamation scheme advances a counter sufficiently. This has the same safety guarantees as
/// guards `unlinked` method, with the added subtlety that there is no reliable ordering for when
/// the destructors corresponding to destroy_and_free will actually be run, only that earlier
/// epochs' destructors will be called first.
unsafe fn destroy_and_free<'a, T>(ptr: Shared<'a, T>, guard: &'a Guard) {
    let ptr = ptr.as_raw() as usize;
    guard.destructor(Box::new(move || {
        drop(Box::from_raw(ptr as *mut T));
    }))
}

/// The inner loop for an operation collating debug information about a `Segment` list
fn dbg_print<'a, 'b: 'a, T: Debug>(a: &'a Atomic<Segment<T>>,
                                   g: &'b Guard)
                                   -> (Option<&'a Atomic<Segment<T>>>, Vec<String>) {
    let s = a.load(Acquire, g).unwrap();
    let addr = a.ptr.load(Acquire) as usize;
    let mut res = Vec::new();
    res.push(format!("Segment id {} <{}>\n", s.id.load(Acquire), addr));
    let mut i = 0;
    while i < SEG_SIZE {
        unsafe {
            let cell = s.data.get_unchecked(i as usize);
            let deleted = cell.stamp.is_deleted(Acquire);
            match cell.data.load(Acquire, g) {
                Some(dat) => {
                    res.push(format!(" (stamp: {:?} deleted: {}, {:?}) ",
                                     cell.stamp,
                                     deleted,
                                     dat))
                }
                None => res.push(format!(" (deleted: {}, None) ", deleted)),
            };
            if i % 4 == 0 {
                res.push("\n".to_string());
            }
        }
        i = i + 1;
    }
    let next_p = s.next.load(Acquire, g);
    let next_id = match next_p {
        Some(_) => s.next.ptr.load(Acquire) as usize,
        None => 0,
    };
    res.push(format!("Next id {}", next_id));
    res.push("\n".to_string());
    (if next_p.is_some() {
         Some(&s.next)
     } else {
         None
     },
     res)
}

/// A utility function for printing out a `Segment`
fn print_loop<'a, T: Debug>(p: &Atomic<Segment<T>>, g: &'a Guard) {
    let mut cur = p;
    loop {
        let (ptr, strings) = dbg_print(cur, g);
        for s in strings {
            print!("{}", s);
        }
        match ptr {
            Some(ptr) => cur = &ptr,
            None => break,
        };
    }
}


impl<K: Eq + Hash + Clone, V> LazySet<K, V> {
    pub fn iter_live<'a>(&self,
                         ord: Ordering,
                         guard: &'a Guard)
                         -> SegCursorLiveData<'a, K, Option<V>> {
        SegCursorLiveData {
            seen: VecSet::new(),
            cursor: self.head.load(ord, guard).unwrap().iter_deref(ord, guard),
        }
    }

    pub fn new() -> LazySet<K, V> {
        let res = LazySet {
            head_ptr: AtomicIsize::new(0),
            head: Atomic::null(),
            delete_ptr: AtomicUsize::new(0),
            // deletion_cache: Atomic::null(),
            cleaning: AtomicBool::new(false),
            die: AtomicBool::new(false), // for debugging
            last_snapshot_len: AtomicUsize::new(0),
            cas_failures: AtomicUsize::new(0),
        };
        res.head.store(new_seg(0), Relaxed);
        res
    }

    /// `search_forward` attempts to find an index starting from head and adding segments on top of
    /// head.
    fn search_forward<'a>(&self,
                          ind: isize,
                          g: &'a Guard,
                          vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>)
                          -> Option<&'a MarkedCell<(K, Option<V>)>> {
        let (seg_id, seg_ind) = split_index(ind);
        // runs at most seg_id - head.id times
        while let Some(seg) = self.head.load(Relaxed, g) {
            let cur_id = seg.id.load(Relaxed);
            if cur_id == seg_id {
                // We have found the segment with the correct id, now load the correct index
                // from the data field.
                unsafe {
                    return Some(seg.data.get_unchecked(seg_ind));
                }
            } else if seg_id < cur_id {
                return None;
            }
            let new_seg = match vec.pop() {
                Some(owned_seg) => {
                    owned_seg.id.store(cur_id + 1, Release);
                    owned_seg.next.store_shared(Some(seg), Release);
                    owned_seg
                }
                None => Owned::new(Segment::new(cur_id + 1, Some(seg))),
            };
            let res = self.head.cas(Some(seg), Some(new_seg), AcqRel);
            if let Err(Some(ptr)) = res {
                vec.push(ptr);
                if DBG {
                    let _ = self.cas_failures.fetch_add(1, Relaxed);
                }
            }
        }
        panic!("programmer error. head should be initialized")
    }

    /// `search_backward` starts with head and follows next pointers looking for the cell at index
    /// ind
    fn search_backward<'a>(&self,
                           ind: isize,
                           g: &'a Guard)
                           -> Option<&'a MarkedCell<(K, Option<V>)>> {
        let (seg_id, seg_ind) = split_index(ind);
        let mut cur = &self.head;
        while let Some(seg) = cur.load(Relaxed, g) {
            let cur_id = seg.id.load(Relaxed);
            if cur_id == seg_id {
                unsafe {
                    return Some(seg.data.get_unchecked(seg_ind));
                }
            }
            cur = &seg.next;
        }
        None
    }
    fn search_kv<'a, 'b: 'a>(&'a self,
                             g: &'b Guard,
                             hash: usize,
                             k: &K)
                             -> LookupResult<&'a MarkedCell<(K, Option<V>)>> {
        for cell in self.head.load(Relaxed, g).unwrap().iter_raw(Relaxed, g) {
            if cell.stamp.matches(hash, Relaxed) {
                if let Some(data) = cell.data.load(Relaxed, g) {
                    if data.0 == *k {
                        return if cell.stamp.is_deleted(Relaxed) {
                            LookupResult::Deleted
                        } else {
                            LookupResult::Found(cell)
                        };
                    }
                }
            }
        }
        return LookupResult::NotFound;
    }

    /// `lookup` returns a reference into `self` that lives as long as `guard`. We get away with
    /// not bounding this in terms of self because we know no references to the returned values
    /// will be freed until `guard` goes out of scope.
    pub fn lookup<'a, 'b: 'a>(&self, key: &'b K, guard: &'a Guard) -> LookupResult<&'a V> {
        let mut h = fnv::FnvHasher::default();
        key.hash(&mut h);
        let hash = h.finish();
        self.lookup_stamped(hash as usize, key, guard)
    }

    /// `lookup` but with a pre-computed stamp.
    pub fn lookup_stamped<'a, 'c, 'b: 'a + 'c>(&'c self,
                                               hash: usize,
                                               key: &'b K,
                                               guard: &'a Guard)
                                               -> LookupResult<&'a V> {

        match self.search_kv(guard, hash, key) {
            LookupResult::Found(cell) => {
                let v_opt = &cell.data.load(Relaxed, guard).unwrap().1;
                let v_unwrap = &v_opt.as_ref()
                    .expect("value shoudl be present for undeleted record");
                LookupResult::Found(v_unwrap)
            }
            LookupResult::Deleted => LookupResult::Deleted,
            LookupResult::NotFound => LookupResult::NotFound,
        }
    }

    /// contains wraps the Segment search method; we linearize this at the moment of the load
    /// operation. Lookup is effectively the same, but it will likely be used directly through
    /// 'search' by higher-level operations.
    pub fn contains(&self, key: K) -> bool {
        let mut h = fnv::FnvHasher::default();
        key.hash(&mut h);
        let hash = h.finish();
        self.contains_stamped(hash as usize, key)
    }

    /// `contains` but with a pre-computed stamp.
    pub fn contains_stamped(&self, hash: usize, key: K) -> bool {
        let guard = epoch::pin();
        self.lookup_stamped(hash, &key, &guard).is_found()
    }

    /// a public `add` function that computes the hash for `key` and passes an empty vector to
    /// `add_stamped`
    pub fn add(&self, key: K, val: V) {
        let mut h = fnv::FnvHasher::default();
        key.hash(&mut h);
        let hash = h.finish();
        let mut v = Vec::new();
        self.add_stamped(hash as usize, key, val, false, &mut v);

    }

    #[inline]
    fn add_stamped(&self,
                   hash: usize,
                   key: K,
                   val: V,
                   growing: bool,
                   vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>) {
        self.add_opt(hash, key, Some(val), false, growing, vec)
    }

    /// add_opt adds `key` and `val` to the front of the bucket with `hash` as the stamp. If
    /// `start_deleted` is true then the record is immediately marked as deleted, which serves as a
    /// removal operation if there is a concurrent grow occurring. `vec` is thread-local data used
    /// to cache segment allocations.
    fn add_opt(&self,
               hash: usize,
               key: K,
               val: Option<V>,
               start_deleted: bool,
               growing: bool,
               vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>) {
        // assuming fetch_add returns old value, we increment head_ptr, look up the cell we
        // acquire, and write K,V to the new cell.
        let guard = epoch::pin();
        let my_ind = self.head_ptr.fetch_add(1, Relaxed);
        let cell = self.search_forward(my_ind, &guard, vec)
            .or_else(|| self.search_backward(my_ind, &guard))
            .expect("error with search"); // neither search succeeds => there is a programmer error
        cell.stamp.init(hash);
        if start_deleted {
            cell.stamp.delete(Acquire, Release);
        }
        cell.data.store(Some(Owned::new((key, val))), Release);
        if DBG {
            if my_ind % 512 == 0 {
                print!("p{} ", my_ind);
                let _ = stdout().flush();

                if my_ind % 8192 == 0 {
                    println!("");
                }
            }
        }

        // every so often, we try and clean up any redundant entries in the bucket. We only call
        // cleanup if we can acquire the `cleaning` flag and if the table isn't growing.
        if growing || (my_ind as usize) & (SEG_CLEANUP_EVERY - 1) != 0 ||
           self.cleaning.compare_and_swap(false, true, Relaxed) {
            return;
        }

        self.cleanup(&guard);
        assert!(self.cleaning.compare_and_swap(true, false, Relaxed));
    }

    /// remove removes `k` from the set if it is present. It returns a boolean indicating whether or
    /// not this particular call to remove was responsible for removing `k` from the data-structure.
    ///
    /// The growing parameter indicates if there is a concurrent growing operation in the hash
    /// table. If that happens, it is possible for there to be a concurrent backfill operation and
    /// we need to ensure that calls to 'contains' see that the removal happened after any adds in
    /// the backfilled nodes.
    ///
    /// Note that remove's return value is best-effort, and it will produce false negatives if
    /// there is a concurrent grow operation (i.e. it is possible for `remove` to return false when
    /// it did in fact delete an element from the data-structure).
    pub fn remove(&self, key: &K, growing: bool) -> bool {
        let mut h = fnv::FnvHasher::default();
        key.hash(&mut h);
        let hash = h.finish();
        let mut v = Vec::new();
        self.remove_stamped(hash as usize, key, growing, &mut v)
    }

    fn remove_stamped(&self,
                      hash: usize,
                      key: &K,
                      growing: bool,
                      vec: &mut Vec<Owned<Segment<(K, Option<V>)>>>)
                      -> bool {
        let guard = epoch::pin();
        if growing {
            // if the table is currently growing, it is possible that we could traverse the list
            // before a backfill completes, thereby failing to delete a node present in the
            // data-structure. To mitigate this remove operations concurrent with a grow operation
            // append a value-less record with key k that is _always_ marked as deleted. Then, we
            // can linearize the removal at the moment the stamp is set, and future search
            // operations will notice a key is present and _will not dereference it because it is
            // marked as deleted_ (this is necessary because we pass uninitialized memory to 'add').
            let copy = key.clone();
            self.add_opt(hash, copy, None, true, growing, vec);
        } else {
            if let LookupResult::Found(n) = self.search_kv(&guard, hash, key) {
                return if !n.stamp.is_deleted(Relaxed) {
                    // TODO we can probably _just_ do delete_fast, because of how lookups work
                    // w.r.t hash collisions (so long as there arent uint64max concurrent remove
                    // operations)
                    n.stamp.delete(Acquire, Release);
                    true
                } else {
                    false
                };
            }
        }
        false
    }

    /// back_fill is a method used to reparent nodes from an old array to a new bucket as part of a
    /// hash table grow or shrink operation. The key algorithm it to read 'v' into a segment list
    /// and then to CAS the list to the end of the current list of segments for 'self'. This ought
    /// to be a scalable operation because only grow/shrink threads have any reason to touch the end
    /// of the list, and there are usually very few of these threads contending for a single bucket.
    ///
    /// `back_fill` (absent bugs) is lock-free as it performs a simple CAS loop on the end of
    /// `self`. Its use in `HashTable` is wait_free as there is only ever one thread performing a
    /// rebucket operation.
    ///
    /// back_fill drains v
    pub fn back_fill(&self, mut v: Vec<(K, V)>) {
        let mut with_hashes = Vec::new();
        while let Some((k, v)) = v.pop() {
            let mut h = fnv::FnvHasher::default();
            k.hash(&mut h);
            let hash = h.finish();
            with_hashes.push((hash as usize, k, v));
        }
        self.back_fill_stamped(with_hashes)
    }

    pub fn back_fill_stamped(&self, mut v: Vec<(usize, K, V)>) {
        if v.is_empty() {
            // nothing to do with empty vector
            return;
        }

        let mut count = -1;        // backwards-moving segment id
        let mut current_index = 0; // index into data array of current Segment
        // Head of our new segment list
        let seg: Segment<(K, Option<V>)> = Segment::new(-1, None);

        let guard = epoch::pin();
        let target_len = v.len();
        let mut i = 0;
        {
            // Current segment being filled
            let mut current_seg = &seg;
            while let Some((h, k, v)) = v.pop() {
                i = i + 1;
                if current_index == SEG_SIZE {
                    count = count - 1;
                    current_index = 0;
                    current_seg.next.store(new_seg(count), Relaxed);
                    current_seg = &*current_seg.next.load(Relaxed, &guard).unwrap();
                }
                unsafe {
                    let cell = current_seg.data.get_unchecked(current_index);
                    cell.data.store(Some(Owned::new((k, Some(v)))), Release);
                    fence(Acquire);
                    cell.stamp.init(h);
                }
                current_index = current_index + 1;
            }
        }
        assert_eq!(target_len, i);
        let mut seg_try = Owned::new(seg);
        // we again assume the head is never null
        let mut cur = self.head.load(Relaxed, &guard).unwrap();
        loop {
            // This loop traverses to the point where a segment next pointer is null, and then
            // attempts a CAS of 'seg' into that next position. The maximum number of iterations
            // the outer loop should perform is the maximum shrinking factor of a single shrinking
            // phase of the outer hash table.
            while let Some(next) = cur.next.load(Relaxed, &guard) {
                cur = next;
            }

            match cur.next.cas_and_ref(None, seg_try, AcqRel, &guard) {
                Ok(_) => {
                    fence(Release);
                    return;
                }
                Err(seg1) => seg_try = seg1,
            };
        }
    }

    /// cleanup searches for adds that have been overwritten and marks them as deleted. Once there
    /// is a segment consisting entirely of deleted nodes, it ensures that the node is unlinked and
    /// marked for deletion. Cleanup also updates `last_snapshot_len`, which is then read by the
    /// `HashTable` code to potentially register a vote for a grow operation.
    ///
    /// `cleanup` is only executed by threads in `add_opt` who successfully CAS `self.clean` to
    /// true.
    pub fn cleanup<'a, 'b: 'a>(&'b self, guard: &'a Guard) {
        let mut local_set = VecSet::new();
        let mut len = 0;
        // fill iterates through a segment's data in reverse order and checks for membership in
        // local_set. If it is a member it means that some value with the same key is ahead of it
        // in the queue, so we mark it as deleted. If it is not a member we add it to the set. If
        // all elements in the segment are deleted then fill returns true, otherwise it returns
        // false.
        fn fill<'a, K: Hash + Eq + Clone, V>(mut local_set: VecSet<K>,
                                             s: Shared<'a, Segment<(K, Option<V>)>>,
                                             len: &mut usize,
                                             g: &'a Guard)
                                             -> (bool, VecSet<K>) {

            let mut all_deleted = true;
            let mut i: i32 = (SEG_SIZE as i32) - 1;
            while i >= 0 {
                let dat = unsafe { s.data.get_unchecked(i as usize) };
                // If we read a null pointer it means that there may be a pending write operation
                // ocurring at that cell. The reason for this is that we only try and clean up
                // segments that are _not_ the head. If s is not the head segment, then a thread
                // must have acquired it in the fetch_add in add. If the cell is still null it
                // means that that thread has yet to complete its add operation, and the cell is
                // not deleted.
                match dat.data.load(Relaxed, g) {
                    Some(d) => {
                        let k = d.0.clone();
                        if local_set.contains(&k) {
                            dat.stamp.delete(Relaxed, Relaxed);
                        } else {
                            if !dat.stamp.is_deleted(Relaxed) {
                                *len = *len + 1;
                                all_deleted = false;
                            }
                            // if it is deleted, but this is the first instance of the key that we
                            // see, we still want to add it to the set because it means all adds
                            // before this one are marked as deleted.
                            local_set.insert(k);
                        }
                    }
                    None => all_deleted = false,
                }
                i = i - 1;
            }
            (all_deleted, local_set)
        }

        // traverse the segment list from the head bacwards, run fill each time, if fill returns
        // true attempt to unlink the segment from the list.
        let mut prev_seg = self.head.load(Relaxed, guard).unwrap();
        let mut cur = &prev_seg.next;
        while let Some(cur_seg_ptr) = cur.load(Relaxed, guard) {
            let (done, set) = fill(local_set, cur_seg_ptr, &mut len, guard);
            local_set = set;
            let next = cur_seg_ptr.next.load(Relaxed, guard);
            if done && prev_seg.next.cas_shared(Some(cur_seg_ptr), next, AcqRel) {
                cur = &cur_seg_ptr.next;
                unsafe {
                    for d in cur_seg_ptr.data.iter() {
                        destroy_and_free(d.data.load(Relaxed, &guard).unwrap(), &guard);
                    }
                    guard.unlinked(cur_seg_ptr);
                }
            } else {
                prev_seg = cur_seg_ptr;
                cur = &cur_seg_ptr.next;
            }
        }
        self.last_snapshot_len.store(len, Relaxed);
    }
}

/// a `MakedCell<T>` contains a pointer to data as well as a `stamp` or hash of a key where the
/// least significant bit denotes if the element has been deleted.
#[derive(Debug)]
struct MarkedCell<T> {
    data: Atomic<T>,
    stamp: Stamp,
}


/// A fixed-length array of `MarkedCell<T>` with a next pointer, implementing a segmented
/// singly-linked list.
#[derive(Debug)]
struct Segment<T> {
    id: AtomicIsize,
    data: [MarkedCell<T>; SEG_SIZE],
    next: Atomic<Segment<T>>,
}

struct SegCursor<'a, T: 'a> {
    ord: Ordering,
    guard: &'a Guard,
    ix: usize,
    cur_seg: Option<Shared<'a, Segment<T>>>,
}

impl<'a, T> Iterator for SegCursor<'a, T> {
    type Item = &'a MarkedCell<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // if current segment is none, we have nothing to do
        match self.cur_seg {
            Some(ptr) => {
                let cell = unsafe { ptr.data.get_unchecked(self.ix) };
                if self.ix == 0 {
                    self.cur_seg = ptr.next.load(self.ord, self.guard);
                    self.ix = SEG_SIZE - 1;
                } else {
                    self.ix -= 1;
                }
                Some(cell)
            }
            None => None,
        }
    }
}

struct SegCursorDerefItems<'a, T: 'a>(SegCursor<'a, T>);

impl<'a, T: 'a> Iterator for SegCursorDerefItems<'a, T> {
    type Item = (&'a Stamp, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(cell) = self.0.next() {
            if let Some(shared) = cell.data.load(self.0.ord, self.0.guard) {
                return Some((&cell.stamp, &*shared));
            }
        }
        return None;
    }
}

pub struct SegCursorLiveData<'a, K: Eq + 'a, V: 'a> {
    seen: VecSet<&'a K>,
    cursor: SegCursorDerefItems<'a, (K, V)>,
}

impl<'a, K: Eq + 'a, V: 'a> Iterator for SegCursorLiveData<'a, K, V> {
    type Item = (usize, &'a (K, V));
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((ref stamp, tup)) = self.cursor.next() {
            let ref k = tup.0;
            if !self.seen.contains(&k) {
                self.seen.insert(k);
                if stamp.is_deleted(self.cursor.0.ord) {
                    continue;
                }
                return Some((stamp.0.load(self.cursor.0.ord), tup));
            }
        }
        return None;
    }
}

/// split_index, given `ind` returns the id of the segment to which `ind` corresponds, along with
/// the index into that segment.
fn split_index(ind: isize) -> (isize, usize) {
    let seg = ind >> SEG_SHIFT;
    let seg_ind = (ind as usize) & (SEG_SIZE - 1);
    (seg, seg_ind)
}

/// helper function for initializing a new `Segment`
fn new_seg<T>(i: isize) -> Option<Owned<Segment<T>>> {
    Some(Owned::new(Segment::new(i, None)))
}

impl<T> Segment<T> {
    fn iter_raw<'a>(&self, ord: Ordering, guard: &'a Guard) -> SegCursor<'a, T> {
        unsafe {
            SegCursor {
                ord: ord,
                guard: guard,
                ix: SEG_SIZE - 1,
                cur_seg: Some(Shared::from_ref(self)),
            }
        }
    }

    fn iter_deref<'a>(&self, ord: Ordering, guard: &'a Guard) -> SegCursorDerefItems<'a, T> {
        SegCursorDerefItems(self.iter_raw(ord, guard))
    }

    /// new allocates a new segment, it initializes the memory in the 'data' member for a segment
    /// with deleted set to false and data set to null.
    fn new<'a>(id: isize, next: Option<Shared<'a, Segment<T>>>) -> Segment<T> {
        let mut seg = Segment {
            id: AtomicIsize::new(id),
            data: unsafe { mem::uninitialized() },
            next: unsafe { mem::uninitialized() },
        };
        for val in seg.data.iter_mut() {
            val.stamp.init(0);
            val.data.store(None, Relaxed);
        }
        seg.next.store_shared(next, Relaxed);
        fence(SeqCst);
        seg
    }
}

unsafe impl<T: Send> Sync for Segment<T> {}
unsafe impl<T: Send> Sync for MarkedCell<T> {}
unsafe impl<K: Send + Eq + Clone + Hash, V: Send> Sync for LazySet<K, V> {}
unsafe impl<K: Send + Eq + Clone + Hash, V: Send> Send for LazySet<K, V> {}

#[cfg(test)]
mod tests {
    extern crate fnv;
    use super::{LazySet, SEG_SIZE, split_index, HashHandle};
    use std::hash::{Hash, Hasher};
    use mem::epoch;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn split_indices() {
        let sz = SEG_SIZE as isize;
        assert_eq!((1, 1), split_index(sz + 1));
        assert_eq!((0, SEG_SIZE - 1), split_index(sz - 1));
        assert_eq!((98, 4), split_index(98 * sz + 4));
    }

    #[test]
    fn basic_functionality() {
        let l: LazySet<usize, usize> = LazySet::new();
        l.add(1, 1);
        l.add(2, 1);
        assert!(l.contains(1));
        assert!(l.contains(2));
        assert!(!l.contains(3));
        l.add(3, 1);
        assert!(l.contains(3));
        l.remove(&3, false);
        assert!(!l.contains(3));
        l.remove(&1, false);
        assert!(!l.contains(1));
        assert!(l.contains(2));
    }
    #[test]
    fn passing_segments_single_threaded() {
        let l = LazySet::new();
        for k in 0..(SEG_SIZE * 3 + 1) {
            l.add(k, k + 1);
        }

        assert!(l.contains(1));
        assert!(l.contains(SEG_SIZE + 1));
        assert!(l.contains(2 * SEG_SIZE + 3));

        for k in SEG_SIZE..(2 * SEG_SIZE) {
            l.remove(&k, false);
        }

        assert!(!l.contains(SEG_SIZE));
        assert!(l.contains(2 * SEG_SIZE + 3));
        let g = epoch::pin();
        l.cleanup(&g);
        assert!(!l.contains(SEG_SIZE));
        assert!(l.contains(2 * SEG_SIZE + 3));
        assert!(l.contains(1));
        l.back_fill((SEG_SIZE * 8..SEG_SIZE * 10)
            .map(|i| (i, i + 1))
            .collect());

        assert!(l.contains(2 * SEG_SIZE + 3));
        assert!(l.contains(1));
        for k in SEG_SIZE * 8..SEG_SIZE * 10 {
            if !l.contains(k) {
                let mut hasher = fnv::FnvHasher::default();
                k.hash(&mut hasher);
                let hash = hasher.finish();
                println!("failing on key {} (hash {} {})",
                         k,
                         hash as usize,
                         hash as usize | 1);
                l.dbg_print_main();
                assert!(l.contains(k));
            }
        }
    }

    #[test]
    fn passing_segments_single_threaded_dcache() {
        let l = LazySet::new();
        for k in 0..(SEG_SIZE * 3 + 1) {
            l.add(k, k + 1);
        }

        assert!(l.contains(1));
        assert!(l.contains(SEG_SIZE + 1));
        assert!(l.contains(2 * SEG_SIZE + 3));

        for k in SEG_SIZE..(2 * SEG_SIZE) {
            l.remove(&k, true);
        }

        for k in SEG_SIZE * 8..SEG_SIZE * 10 {
            l.remove(&k, true);
        }

        for k in SEG_SIZE * 8..SEG_SIZE * 10 {
            assert!(!l.contains(k));
        }

        l.back_fill((SEG_SIZE * 9..SEG_SIZE * 10)
            .map(|i| (i, i + 1))
            .collect());

        for k in SEG_SIZE * 8..SEG_SIZE * 10 {
            if l.contains(k) {
                println!("failing on {}", k);
                l.dbg_print_main();
                assert!(!l.contains(k));
            }
        }


        assert!(!l.contains(SEG_SIZE));
        assert!(l.contains(2 * SEG_SIZE + 3));
        let g = epoch::pin();
        l.cleanup(&g);
        assert!(!l.contains(SEG_SIZE));
        assert!(l.contains(2 * SEG_SIZE + 3));
        assert!(l.contains(1));
    }

    #[test]
    fn concurrent_enqueues_add_only_f() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        let l = Arc::new(LazySet::new());
        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    l.add(j, j);
                    l.remove(&(j + perthread), false);
                    l.remove(&2, false);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }

        for i in 0..perthread {
            if i == 2 {
                assert!(!l.contains(i));
            } else {
                assert!(l.contains(i));
            }
        }
    }

    #[test]
    fn concurrent_enqueues_add_only_dcache() {
        // bumping this up higher makes the test take a long time. I suspect it has to do with the
        // deletion cache growing too large, making the algorithm quadratic. Check back once
        // cleanup also compresses deletion caches.
        let nthreads = 8; //64;
        let perthread = 2048;
        let mut v = Vec::new();
        let l = Arc::new(LazySet::new());
        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    l.add(j, j);
                    l.remove(&(j + perthread), true);
                    l.remove(&2, true);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }
        for i in 0..perthread {
            if i == 2 {
                assert!(!l.contains(i));
            } else {
                assert!(l.contains(i));
            }
        }
    }

    #[test]
    fn concurrent_ops_del_cache() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        // note that patterns like this will leak memory, but we are just checking functional
        // correctness here.
        let l = Arc::new(LazySet::new());
        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    l.remove(&j, true);
                    if j % 100 == 0 {
                        l.back_fill((j..perthread + j)
                            .map(|x| (x, x + 1))
                            .collect());
                    }
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => {
                    println!("Thread panicked! marking test as failed");
                    assert!(false);
                }
            };
        }

        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    l.add(j, j + 1);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => {
                    println!("Thread panicked! marking test as failed");
                    assert!(false);
                }
            };
        }
    }

    #[test]
    fn concurrent_ops_all_bucket_str() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        let l = Arc::new(LazySet::new());
        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in (0..perthread).map(|num| format!("{}", num)) {
                    let v = j.clone();
                    l.add(j, v);
                    let _ = l.remove(&"2".to_string(), true);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => {
                    println!("Thread panicked! marking test as failed");
                    assert!(false);
                }
            };
        }
    }

    #[test]
    fn concurrent_ops_all_bucket() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        let l = Arc::new(LazySet::new());
        for _i in 0..nthreads {
            let l = l.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    l.add(j, j);
                    l.remove(&(j + perthread), true);
                    l.remove(&(j + j), true);
                    if j % 100 == 0 {
                        l.back_fill((j..perthread + j)
                            .map(|x| (x, x + 1))
                            .collect());
                    }
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => {
                    println!("Thread panicked! marking test as failed");
                    assert!(false);
                }
            };
        }
    }

    #[test]
    fn concurrent_ops_remove_2_ht() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        let table = HashHandle::new();
        for _i in 0..nthreads {
            let mut t = table.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    t.insert(j, j + 1);
                    t.remove(&2);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }
        table.with_table(|l| {
            for i in 0..perthread {
                if i == 2 {
                    assert!(!l.contains(&i));
                } else {
                    assert!(l.contains(&i));
                    assert!(l.scoped_lookup(&i, |o| {
                        match o {
                            Some(n) => (*n) == i + 1,
                            None => false,
                        }
                    }));
                }
            }
        });
    }
    #[test]
    fn concurrent_ops_all_ht() {
        let nthreads = 64;
        let perthread = 2048;
        let mut v = Vec::new();
        let t = HashHandle::new();
        for _i in 0..nthreads {
            let mut t = t.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    t.insert(j, j);
                    t.remove(&(j + perthread));
                    t.remove(&(j + j));
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }
    }

    #[test]
    fn concurrent_ops_all_ht_grow() {
        let nthreads = 64;
        let perthread = 8192;
        let mut v = Vec::new();
        let table = HashHandle::with_capacity_grow(2, 4);
        for _i in 0..nthreads {
            let mut t = table.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    t.insert(j, j);
                    t.remove(&(j + perthread));
                    t.remove(&(j + j));
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }
    }

    #[test]
    fn concurrent_remove_ht_grow() {
        let nthreads = 64;
        let perthread = 8192;
        let mut v = Vec::new();
        let table = HashHandle::with_capacity_grow(2, 4);
        for _i in 0..nthreads {
            let mut t = table.clone();
            v.push(thread::spawn(move || {
                for j in 0..perthread {
                    t.insert(j, j);
                    t.remove(&2);
                }
            }));
        }

        while let Some(handle) = v.pop() {
            match handle.join() {
                Ok(()) => continue,
                Err(_) => assert!(false),
            };
        }

        let mut all_good = true;
        table.with_table(|l| {
            for i in 0..perthread {
                if i == 2 {
                    assert!(!l.contains(&i));
                } else {
                    if !l.scoped_lookup(&i, |o| {
                        match o {
                            Some(n) => (*n) == i,
                            None => {
                                let mut hasher = fnv::FnvHasher::default();
                                i.hash(&mut hasher);
                                println!("couldn't find {}, (hash {})", i, hasher.finish());
                                false
                            }
                        }
                    }) {
                        all_good = false;
                    }
                    assert!(l.contains(&i));
                }
            }
        });
        assert!(all_good);
    }
}

/// Some debugging routines for LazySet, these print and check some invariants.
impl<K: Eq + Hash + Clone + Debug, V: Debug> LazySet<K, V> {
    // prints detailed information about the main path of a segment,
    pub fn dbg_print_main(&self) {
        let guard = epoch::pin();
        println!("Main seglist:");
        print_loop(&self.head, &guard);
    }

    pub fn check_decreasing<F>(&self, msg: F)
        where F: Fn()
    {
        let guard = epoch::pin();
        let mut cur = self.head.load(Relaxed, &guard);
        let mut prev_seg = isize::MAX;
        while let Some(seg) = cur {
            let s_id = seg.id.load(Relaxed);
            if prev_seg >= 0 && prev_seg <= s_id {
                msg();
            }
            prev_seg = s_id;
            cur = seg.next.load(Relaxed, &guard);
        }
    }

    pub fn consistency_check(&self) {
        let guard = epoch::pin();
        let mut h = VecSet::new();
        let mut v = Vec::new();
        let mut shadow = Vec::new();
        let mut cur_seg = &self.head;
        loop {
            let ptr = cur_seg.ptr.load(Relaxed) as usize;
            if ptr == 0 {
                break;
            }
            let (_next, snapshot) = dbg_print(cur_seg, &guard);
            if h.contains(&ptr) {
                shadow.push((ptr, snapshot.clone()));
            } else {
                h.insert(ptr);
            }
            v.push((ptr, snapshot));
            if shadow.len() > 10 {
                break;
            }
            match cur_seg.load(Relaxed, &guard) {
                Some(seg) => cur_seg = &seg.next,
                None => break,
            }
        }
        if shadow.len() > 10 {
            if !self.die.compare_and_swap(false, true, Relaxed) {
                let s: Vec<String> = shadow.iter().map(|t| format!("{}", t.0)).collect();
                println!("Saw the following multiple times: {}", s.join(","));
                println!("What this check saw:");
                for (_i, vec) in v {
                    println!("{}", vec.join(""));
                }
            }
        }
    }
}

impl<K: Eq + Hash + Clone + Debug, V: Clone + Debug> HashTable<K, V> {
    /// debugging routine that prints the internal layout of `self`.
    pub fn dbg_print_main(&self) {
        let guard = epoch::pin();
        let print_bucket_array = |buckets: &Atomic<Vec<LazySet<K, V>>>| {
            for (i, bucket) in buckets.load(Relaxed, &guard).unwrap().iter().enumerate() {
                println!("bucket {}", i);
                bucket.dbg_print_main();
            }
        };
        println!("main array, len {}",
                 self.buckets.load(Relaxed, &guard).unwrap().len());
        print_bucket_array(&self.buckets);
        print!("prev array, ");
        if let Some(prev_buckets) = self.prev_buckets.load(Relaxed, &guard) {
            println!("len {}", prev_buckets.len());
            print_bucket_array(&self.prev_buckets);
        } else {
            println!("(None)");
        }
    }
}


