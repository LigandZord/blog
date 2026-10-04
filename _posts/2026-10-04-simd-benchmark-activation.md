---
title: "SIMD SiLU implementation and its comparison with torch"
date: 2026-10-04 12:00:00 +0000
categories: [Deep Learning in Rust, Activation]
tags: [rust, simd, avx2, benchmarking]
math: true
---

## Recap

In the previous post, we saw the naive implementation of the activation function, which is noticeably slower than the torch implementation, as well as slower when we enable multi-threading in torch. In this post, I explore the SIMD implementation of SiLU activation in Rust, and benchmark it against both torch and the naive Rust implementation.

## Theory

To first learn more about SIMD and its implementation in Rust, I explored a few YouTube videos. The one that helped me best is "How to use SIMD (x86_64) in Rust" by Andres Quintero. From there, I understood that there are some Intel functions that I must directly use to load the required data from memory, and that only some operations are available directly through these: things like addition, multiplication, subtraction, division, etc. But for our SiLU implementation, we need the exponential function, which is unfortunately unavailable in the AVX2 functions that Intel offers.

To get around this, I went and explored a bit of theory on approximating the exponential function using the basic math operations we do have available. I came across the Schraudolph fast approximation for getting the value of `e^x`. But I also learnt that this is a crude approximation, and not what's actually used in current systems. I then came across the Cephes/Pommier algorithm for this. This was extremely interesting — the basis of it is that for a computer, shifting bits is an extremely fast operation compared to actually doing the math. So how do we exploit this? What if we convert the exponential into a power of 2? That's exactly what this algorithm does. Here's the basic math of it:

We can write:

$$e^x = 2^{x \cdot \log_2(e)}$$

So we need to compute $t = x \cdot \log_2(e)$. `log2(e)` is a constant we only need to calculate once — in Rust, `std::f32::consts::LOG2_E` gives us exactly this.

Now, the issue is that `t` isn't an integer, and bit shifting only works on integer values. So we split `t` into two parts: an integer part `n` and a remainder `r`:

$$t = n + r, \quad r \in [-0.5, 0.5]$$

And hence:

$$e^x = 2^t = 2^{n+r} = 2^n \cdot 2^r$$

The `2^n` part can be computed extremely fast using bit shifting. The `2^r` part is approximated using a Taylor expansion — in my code, I approximated it using 6 terms.

Using this approach, I could approximate `e^x` using bit manipulation and a Taylor expansion — just addition and multiplication of numbers, both of which are available in AVX2.

## Some things that I noticed

There were some nitty-gritty details I noticed before actually implementing this. For example, what I've implemented is for f32 precision only, so I had to understand how a 32-bit float is stored in memory. Out of the 32 bits, 1 is a sign bit, 8 are exponent bits, and the other 23 are mantissa bits. To get the `2^n` value, I had to find out how the exponent bits represent negative and positive values for a 32-bit floating point number — for this, a bias term of 127 (specific to f32) needs to be added to the value of `n`.

Another important thing is to clamp the values of `x` before actually doing the bit shifting. This is required because we need to make sure whatever value of `n` we get stays within the range the f32 exponent bits can actually represent. If the input is very large, we end up with junk values instead of the `inf`/`-inf` we'd expect. It was pretty interesting to look at these system-level details as I implemented this — which was the whole point of the exercise.

## System configuration

Same machine as the previous post, for reference:

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H — 11 cores / 22 threads (hyperthreading), 1 socket |
| Instruction sets | AVX2, FMA, AVX-VNNI (no AVX-512) |
| RAM | 7.4 GiB (WSL2 VM allocation) |
| OS | Ubuntu 26.04.1 LTS on WSL2, kernel 6.18.40.1-microsoft-standard-WSL2 |
| Rust | rustc / cargo 1.98.1 |
| PyTorch | 2.14.0+cpu, numpy 2.5.3 |

## Benchmark

Before trusting any speed numbers, I made sure `silu_simd` was actually correct: checked against `std::f32::exp` directly (max relative error ~6.6e-6), against the naive implementation (max abs error ~4.8e-7), and against PyTorch's real output on 10,000 random fixtures spanning well past the clamp boundary (max abs error ~6e-7, zero outside tolerance). With that settled, here's how it performs on the same 1,048,576 `f32` elements as before:

| Implementation | Time/iter | Throughput |
|---|---|---|
| `silu_naive` (Rust, scalar) | 2.62 ms | 0.40 G-elem/s |
| `silu_simd` (Rust, AVX2+FMA) | 401 µs | 2.61 G-elem/s |
| torch CPU, 1 thread | 1011 µs | 1.04 G-elem/s |
| torch CPU, 11 threads (default) | 589 µs | 1.78 G-elem/s |

`silu_simd` is about 6.5x faster than my own naive baseline, and - the number I'm most happy with - about 2.5x faster than torch's single-threaded kernel, and roughly 1.5x faster than torch's default run using all 11 threads. A single thread of hand-written AVX2 is beating PyTorch's fully parallel CPU kernel.

## What's next

The next stage in this series is multithreading this kernel with `rayon`, which should widen that gap even further. After that, it's on to Linear/GEMM - the highest-leverage kernel in this whole project.
