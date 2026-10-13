// C entry points so the Python fitting/validation uses the exact plugin DSP code.
#include "../dsp/kobol_vcf.h"
extern "C" {
// render a block with constant (or per-sample, if fc_arr != 0) parameters
void vcf_render(const double* x, double* out, int n, double fs, double fc, double K,
                double drive, double gain, int iters, const double* fc_arr, double q) {
    kobol::SSM2040 f; f.setSampleRate(fs); f.iterations = iters; f.fbKnee = q;
    for (int i = 0; i < n; ++i) out[i] = f.process(x[i], fc_arr ? fc_arr[i] : fc, K, drive, gain);
}
}
#include "../dsp/kobol_vcf_law.h"
#include "../dsp/kobol_env_vca.h"
extern "C" {
void vcf_law(double cutoffV, double resV, double* out) {
    auto p = kobol::vcflaw::evaluate(cutoffV, resV); out[0] = p.fc; out[1] = p.K; out[2] = p.gainDb;
}
// ADS -> VCA test render: gate[i] in {0,1}
void env_vca_render(const double* x, const double* gate, double* env, double* out, int n, double fs,
                    double a, double d, double s) {
    kobol::SSM2050 e; e.setSampleRate(fs); kobol::CA3080VCA v; v.setSampleRate(fs);
    for (int i = 0; i < n; ++i) { env[i] = e.process(gate[i] > 0.5, a, d, s); out[i] = v.process(x[i], env[i]); }
}
}
#include "../dsp/kobol_voice.h"
#include <chrono>
extern "C" {
// note held from 0.05 s to 70% of the render; returns seconds of CPU time
double voice_render(double* out, int n, double fs, int note, double cutoffV, double resV, double envAmt,
                    double a1, double d1, double s1, double a2, double d2, double s2) {
    kobol::Voice v; v.prepare(fs); kobol::VoiceParams p;
    p.cutoffV = cutoffV; p.resV = resV; p.envAmountV = envAmt;
    p.a1 = a1; p.d1 = d1; p.s1 = s1; p.a2 = a2; p.d2 = d2; p.s2 = s2;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        if (i == (int) (0.05 * fs)) v.noteOn(note);
        if (i == (int) (0.7 * n)) v.noteOff();
        out[i] = v.process(p);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}
