# Mojo VS JAX
I was a bit confused when Mojo showed only 0.7x the speed vs numpy on my machine when I thought they were promising huge speedups on their website.  
So I thought, lets test Mojo against JAX (I'm no expert and only began learning JAX recently, so it's just very basic JIT compilation and jax.numpy) and see how it performs.  

All I did was add a few benchmark functions and run them in the matmul.mojo from the official mojo examples (see: https://github.com/modularml/mojo.git).

### Requirements
You just need to have Mojo installed (see: https://docs.modular.com/mojo/manual/get-started/)  
and Python with JAX installed (see: https://github.com/google/jax).  

### Results
I'm running an i7 4710HQ and a GTX 970M in my nearly 10 year old laptop and this was what I got as output:  


    CPU Results  
      
    Python:         0.002 GFLOPS  
    Numpy:        279.042 GFLOPS  
    JNP:          206.906 GFLOPS  
    JNP JIT:     1006.997 GFLOPS  
    Naive:          3.100 GFLOPS   1692.29x Python  0.01x Numpy  
    Vectorized:     9.888 GFLOPS   5398.10x Python  0.04x Numpy  
    Parallelized:  22.155 GFLOPS  12094.57x Python  0.08x Numpy  
    Tiled:         18.167 GFLOPS   9917.22x Python  0.07x Numpy  
    Unrolled:      15.889 GFLOPS   8673.91x Python  0.06x Numpy  
    Accumulated:  206.506 GFLOPS 112732.49x Python  0.74x Numpy    
      
      
    Naive:          2.912 GFLOPS      0.01x Python  0.00x Numpy  
    Vectorized:    10.097 GFLOPS      0.05x Python  0.01x Numpy  
    Parallelized:  22.259 GFLOPS      0.11x Python  0.02x Numpy  
    Tiled:         17.975 GFLOPS      0.09x Python  0.02x Numpy  
    Unrolled:      15.923 GFLOPS      0.08x Python  0.02x Numpy  
    Accumulated:  206.417 GFLOPS      1.00x Python  0.20x Numpy  

In the second repetition of comparisons, `Python` equals `jax.numpy` and `Numpy` stands for the `jax.numpy + jit` version of matrix multiplication (i was too lazy to change the strings again).  
As is visible, JAX+JIT outperforms Mojo by far, which was honestly not what I expected after hearing about Mojo (sadly).  
