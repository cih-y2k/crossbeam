extern crate crossbeam;
extern crate time;
extern crate rand;

use crossbeam::sync::hash_table::HashHandle;
use crossbeam::mem::epoch;
use std::thread;
use std::time::Duration;
use std::env;
use rand::distributions::{IndependentSample, Range};

const RANGE: usize = !0;

macro_rules! bench_inst {
    ($table:expr, $ix:expr, $inputs:expr, insert) => {
        unsafe {
            $table.insert(*$inputs.get_unchecked($ix), *$inputs.get_unchecked($ix))
        }
    };
    ($table:expr, $ix:expr, $inputs:expr, remove) => {
        unsafe {
            $table.remove(&$inputs.get_unchecked($ix))
        }
    };
    ($table:expr, $ix:expr, $inputs:expr, lookup) => {
        unsafe {
            $table.contains(&$inputs.get_unchecked($ix))
        }
    };
}

macro_rules! bench_sched {
    ($table:expr, $ix:expr, $inputs:expr, $op:tt) => {
        bench_inst!($table, $ix, $inputs, $op);
        $ix += 1;
    };
    ($table:expr, $ix:expr, $inputs:expr, $head:tt, $($tail_ops:tt),+) => {
        bench_inst!($table, $ix, $inputs, $head);
        $ix += 1;
        bench_sched!($table, $ix, $inputs, $($tail_ops),*);
    };
}

macro_rules! sched_ratio {
    ($table:expr, $ix:expr, $inputs:expr, 0) => {
        bench_sched!($table, $ix, $inputs, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 10) => {
        bench_sched!($table, $ix, $inputs, insert, remove, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 20) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 30) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, lookup,
                     lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 40) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, lookup, lookup, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 50) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, lookup, lookup, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 60) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, insert, remove, lookup, lookup, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 70) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, insert, remove, insert, remove, lookup, lookup,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 80) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, insert, remove, insert, remove, insert, remove,
                     lookup, lookup, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 90) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, insert, remove, insert, remove, insert, remove,
                     insert, remove, lookup, lookup)
    };

    ($table:expr, $ix:expr, $inputs:expr, 100) => {
        bench_sched!($table, $ix, $inputs, insert, remove, insert, remove, insert, remove, insert,
                     remove, insert, remove, insert, remove, insert, remove, insert, remove,
                     insert, remove, insert, remove)
    }
}

macro_rules! declare_bench {
    ($name:tt, $op:tt) => {
        #[allow(unused_mut)]
        fn $name(nthreads: usize, perthread: usize,
                 table: &HashHandle<usize, usize>,
                 seqs: &Vec<Vec<usize>>) -> u64 {
                let start = time::precise_time_ns();
                crossbeam::scope(|scope| {
                    for i in 0..nthreads {
                        let mut t = table.clone();
                        let my_slice = seqs[i].as_slice();
                        scope.spawn(move || {
                            let mut ctr = 0;
                            for _j in 0..perthread {
                                sched_ratio!(t, ctr, my_slice, $op);
                            }
                        });
                    }
                });
                start
            }
    };
}

declare_bench!(bench_0, 0);
declare_bench!(bench_10, 10);
declare_bench!(bench_20, 20);
declare_bench!(bench_30, 30);
declare_bench!(bench_40, 40);
declare_bench!(bench_50, 50);
declare_bench!(bench_60, 60);
declare_bench!(bench_70, 70);
declare_bench!(bench_80, 80);
declare_bench!(bench_90, 90);
declare_bench!(bench_100, 100);

fn main() {
    let total = 400 * (1 << 10);
    let n_trials = 7;


    // very smart command-line parsing
    let mut iter = env::args();
    let _ = iter.next().unwrap();
    let threads = iter.next()
        .expect("must specify num threads")
        .parse::<usize>()
        .expect("num threads must be integer");

    let prefill_weight = iter.next()
        .expect("must specify prefill weight")
        .parse::<usize>()
        .expect("prefill weight must be integer");

    let percent_mutate = iter.next()
        .expect("must specify percent mutate")
        .parse::<usize>()
        .expect("percent mutate must be integer");

    let mut start_size = 16 * 4096;
    if percent_mutate > 20 {
        start_size *= 8;
    }
    let bench_fn = match percent_mutate {
        0 => bench_0,
        10 => bench_10,
        20 => bench_20,
        30 => bench_30,
        40 => bench_40,
        50 => bench_50,
        60 => bench_60,
        70 => bench_70,
        80 => bench_80,
        90 => bench_90,
        100 => bench_100,
        _ => panic!("percent_mutate must be a multiple of 10 in [0, 100]"),
    };

    let per = total / threads;
    let total = per * threads;
    let per_round = 20;
    let num_ops = total * per_round;
    let mut results: Vec<u64> = Vec::new();
    let mut rng = rand::thread_rng();
    let between = Range::new(0, RANGE);
    println!("generating inputs...");
    let seqs: Vec<Vec<usize>> = (0..threads)
        .map(|_tid| {
            (0..total * per_round)
                .map(|_i| between.ind_sample(&mut rng))
                .collect()
        })
        .collect();
    println!("done");
    for _i in 0..n_trials {
        let table = HashHandle::with_capacity_grow(1, start_size);
        if prefill_weight > 0 {
            let prefill = start_size / prefill_weight;
            println!("prefilling");
            {
                let mut table = table.clone();
                for _j in 0..prefill {
                    let item = between.ind_sample(&mut rng);
                    table.insert(item, item);
                }
            }
            for _i in 0..10 {
                epoch::pin().quiesced_collect();
            }
            println!("done");
        }
        let start = bench_fn(threads, per, &table, &seqs);
        let dur = time::precise_time_ns() - start;
        for _i in 0..10 {
            epoch::pin().quiesced_collect();
        }
        println!("ending after {}", dur);
        results.push(dur);
        thread::sleep(Duration::new(3, 0));
    }
    results.sort();
    let mut total_dur: u64 = 0;
    for i in 1..(results.len() - 1) {
        total_dur += results[i];
    }
    println!("ConcurrentHT-n{}-{}\t\t{} ns ({} ops/s)",
             threads,
             num_ops,
             total_dur / ((results.len() - 2) as u64),
             (num_ops as f64) * ((n_trials - 2) as f64) * 1e9 / (total_dur as f64));
}
