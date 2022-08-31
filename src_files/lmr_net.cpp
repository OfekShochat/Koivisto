#include <algorithm>

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
    void forward(int inputs[16], float out[O]) {
        float scratch[H] = hidden_bias_;
        float scratch2[O] = out_bias_;

        for (int i = 0; i < I; i++)
            dot<I>(inputs, hidden_ + i * I, scratch + i);
        for (int i = 0; i < O; i++)
            dot<H>(scratch, out_ + i * H, scratch2 + i);

        memcpy(out, scratch2);
    }
};
