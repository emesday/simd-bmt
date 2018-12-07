```
$ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
$ uname -a
Darwin **************************** 17.7.0 Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT 2018; root:xnu-4570.71.2~1/RELEASE_X86_64 x86_64
$ make
...
axpy_simd over axpy -38%
axpy_simd over axpy_simdu -31%
dot_simd over dot 91%
dot_simd over dot_simdu 6.3%
...
```

```
$ lscpu | grep "Model name"
Model name:            Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz
$ uname -a
Linux ********************* 3.10.0-862.14.4.el7.x86_64 #1 SMP Wed Sep 26 15:12:11 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
$ make
...
axpy_simd over axpy 33%
axpy_simd over axpy_simdu 30%
dot_simd over dot 94%
dot_simd over dot_simdu 15%
...
```
