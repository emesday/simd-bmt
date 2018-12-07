#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cassert>
#include <vector>

#include <x86intrin.h>

void axpy(const float a, const float* x, float* y, int dim) {
  for (; dim > 0; dim--, x++, y++) {
    *y += a * *x;
  }
}

void axpy_simd(const float a, const float* x, float* y, int dim) {
  __m256 a0 = _mm256_set1_ps(a);
  for (; dim > 0; dim -= 8, x += 8, y += 8) {
    __m256 d = _mm256_mul_ps(a0, _mm256_load_ps(x));
    d = _mm256_add_ps(d, _mm256_load_ps(y));
    _mm256_store_ps(y, d);
  }
}

void axpy_simdu(const float a, const float* x, float* y, int dim) {
  __m256 a0 = _mm256_set1_ps(a);
  for (; dim > 0; dim -= 8, x += 8, y += 8) {
    __m256 d = _mm256_mul_ps(a0, _mm256_loadu_ps(x));
    d = _mm256_add_ps(d, _mm256_loadu_ps(y));
    _mm256_storeu_ps(y, d);
  }
}

float hsum256_ps_avx(__m256 v) {
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1),
      _mm256_castps256_ps128(v));
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
}

float dot(const float* x, const float* y, int dim) {
  float result = 0;
  for (; dim > 0; dim--, x++, y++) {
    result += *x * *y;
  }
  return result;
}

float dot_simd(const float* x, const float* y, int dim) {
  float result = 0;
  __m256 d = _mm256_setzero_ps();
  for (; dim > 7; dim -= 8, x += 8, y += 8) {
    d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_load_ps(x), _mm256_load_ps(y)));
  }
  return hsum256_ps_avx(d);
}

float dot_simdu(const float* x, const float* y, int dim) {
  float result = 0;
  __m256 d = _mm256_setzero_ps();
  for (; dim > 7; dim -= 8, x += 8, y += 8) {
    d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
  }
  return hsum256_ps_avx(d);
}

void zero(float *x, int dim) {
  for (; dim > 0; dim--, x++) {
    *x = 0;
  }
}

int main(int argc, char *argv[]) {
  int size = 8 * 100;
  std::cout << "size " << size << std::endl;
  float *ax = (float*)_mm_malloc(sizeof(float) * size, 64);
  float *ay = (float*)_mm_malloc(sizeof(float) * size, 64);
  float *ux = (float*)malloc(sizeof(float) * size);
  float *uy = (float*)malloc(sizeof(float) * size);

  std::cout << "initialization" << std::endl;
  std::minstd_rand rng(1);
  std::uniform_real_distribution<float> uniform(-1, 1);
  for (int i = 0; i < size; i++) {
    ax[i] = ux[i] = uniform(rng);
  }

  std::cout << "check if each operation works" << std::endl;
  zero(uy, size);
  axpy(1, ux, uy, size);
  for (int i = 0; i < size; i++) {
    assert(ux[i] == uy[i]);
  }
  std::cout << "  axpy passed" << std::endl;
  zero(ay, size);
  axpy_simd(1, ax, ay, size);
  for (int i = 0; i < size; i++) {
    assert(ax[i] == ay[i]);
  }
  std::cout << "  axpy_simd passed" << std::endl;
  zero(uy, size);
  axpy_simdu(1, ux, uy, size);
  for (int i = 0; i < size; i++) {
    assert(ux[i] == uy[i]);
  }
  std::cout << "  axpy_simdu passed" << std::endl;
  float d0 = dot(ux, ux, size);
  std::cout << "  dot(ux, ux) = " << d0 << std::endl;
  float d1 = dot_simd(ax, ax, size);
  std::cout << "  dot_simd(ax, ax) = " << d1 << std::endl;
  assert(std::abs(d0 - d1) < 1e-4);
  float d2 = dot_simdu(ux, ux, size);
  std::cout << "  dot_simdu(ux, ux) = " << d2 << std::endl;
  assert(std::abs(d0 - d2) < 1e-4);
  std::cout << "all passed" << std::endl;

  std::vector<double> elapsed;
  int trial = 10000000;
  std::cout << "bmt" << std::endl;
  {
    zero(uy, size);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      axpy(0.00001, ux, uy, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  axpy elapsed " << *(elapsed.end() - 1) << std::endl;
  }

  {
    zero(ay, size);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      axpy_simd(0.00001, ax, ay, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  axpy_simd elapsed " << *(elapsed.end() - 1) << std::endl;
  }

  for (int i = 0; i < size; i++) {
    assert(std::abs(uy[i] - ay[i]) < 1e-4);
  }

  {
    zero(uy, size);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      axpy_simdu(0.00001, ux, uy, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  axpy_simdu elapsed " << *(elapsed.end() - 1) << std::endl;
  }

  for (int i = 0; i < size; i++) {
    assert(std::abs(uy[i] - ay[i]) < 1e-4);
  }

  trial = 1000000;
  {
    float empty = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      empty *= dot(ux, ux, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  dot elapsed " << *(elapsed.end() - 1) << empty << std::endl;
  }

  {
    float empty = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      empty *= dot_simd(ax, ax, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  dot_simd elapsed " << *(elapsed.end() - 1) << empty << std::endl;
  }

  {
    float empty = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < trial; i++) {
      empty *= dot_simdu(ux, ux, size);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsed.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    std::cout << "  dot_simdu elapsed " << *(elapsed.end() - 1) << empty << std::endl;
  }

  std::cout << "axpy_simd over axpy " << std::setprecision(2) << 100 * (elapsed[0] - elapsed[1]) / elapsed[0] << "%" << std::endl;
  std::cout << "axpy_simd over axpy_simdu " << std::setprecision(2) << 100 * (elapsed[2] - elapsed[1]) / elapsed[2] << "%" << std::endl;
  std::cout << "dot_simd over dot " << std::setprecision(2) << 100 * (elapsed[3] - elapsed[4]) / elapsed[3] << "%" << std::endl;
  std::cout << "dot_simd over dot_simdu " << std::setprecision(2) << 100 * (elapsed[5] - elapsed[4]) / elapsed[5] << "%" << std::endl;

  std::cout << "release resources" << std::endl;
  _mm_free(ax);
  _mm_free(ay);
  free(ux);
  free(uy);
}
