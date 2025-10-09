public class ProofOfConcept {

    private float w1 = 1;
    private float w2 = 1;

    private float lr = 0.01f;
    private float target_err = 0.05f;

    private float ReLu(float x) {
        return Math.max(0, x);
    }

    private float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp((double) -x)));
    }

    private float sigmoid_deriv(float x) {
        return x * (1 - x);
    }

    private float mse(float pred, float actual) {
        return (pred - actual) * (pred - actual);
    }

    public float run(float x) {
        x = this.sigmoid(x * this.w1);
        x = this.sigmoid(x * this.w2);
        return x;
    }

    private float[] runInfo(float x) {
        float[] info = new float[2];
        info[0] = this.sigmoid(x * this.w1);
        info[1] = this.sigmoid(info[0] * this.w2);
        return info;
    }

    public void backpropigate(float x, float y) {
        float[] info = this.runInfo(x);

        float err_2 = this.mse(info[1], y);
        float delta_2 = err_2 * this.sigmoid_deriv(info[1]);

        float err_1 = delta_2 * this.w2;
        float delta_1 = err_1 * this.sigmoid_deriv(info[0]);

        this.w2 += info[0] * delta_2 * this.lr;
        this.w1 += x * delta_1 * this.lr;
    }

    public float fit(float x, float y) {
        float current_err = this.mse(this.run(x), x);
        int i = 0;

        while (current_err > this.target_err && i < 10_000) {
            this.backpropigate(x, y);
            current_err = this.mse(this.run(x), x);
            i++;
        }
        return current_err;
    }

}
