#include <algorithm>
#include <cstring>
#include <iostream>

template <int S>
void dot(float a[S], float b[S], float *out) {
    for (int i = 0; i < S; i++) {
        *out += a[i] * b[i];
    }
}

template <int S>
void relu(float a[S], float out[S], float M = 0.0f, float X = -1.0f) {
    for (int i = 0; i < S; i++) {
        out[i] = std::max(out[i], M);
        if (X > M)
            out[i] = std::min(out[i], X);
    }
}

template <int I, int H, int O>
class Network {
  private:
    alignas(64) float hidden_[I * H];
    alignas(64) float out_[H * O];
    float hidden_bias_[H];
    float out_bias_[O];

  public:
    Network() {
        std::cout << "poop\n";
        #include "../poop.h"
        std::memcpy(hidden_, parameters, I * H);
        std::memcpy(hidden_bias_, parameters + I * H, H);
        std::memcpy(hidden_, parameters + I * H + H, H * O);
        std::memcpy(hidden_, parameters + I * H + H + H * O, O);
    }

    void forward(int inputs[I], float out[O]) {
        float some[I];
        for (int i = 0; i < I; i++) {
            some[i] = (float)inputs[i];
        }

        float scratch[H];
        memcpy(scratch, hidden_bias_, H);
        float scratch2[O];
        memcpy(scratch2, out_bias_, O);

        for (int i = 0; i < I; i++)
            dot<I>(some, hidden_ + i * I, scratch + i);
        for (int i = 0; i < O; i++)
            dot<H>(scratch, out_ + i * H, out + i);
    }
};
