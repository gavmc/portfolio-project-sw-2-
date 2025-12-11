import components.ANN.ANN;
import components.ANN.ANN1L;

public class BinaryCalssifierExample {

    // really not going to work that well, but if you need a proper ML model go learn tensorflow or pytorch

    private ANN[] anns;
    private int numInputs;

    public BinaryCalssifierExample(int numInputs) {
        this.numInputs = numInputs;
        this.anns = new ANN[numInputs];

        for (int i = 0; i < numInputs; i++) {
            this.anns[i] = new ANN1L();
        }
    }

    public float run(float[] x) {
        assert x.length == this.numInputs : "Input length does not match";

        float out = 0;
        for (int i = 0; i < this.numInputs; i++) {
            out += this.anns[i].run(x[i]);
        }

        return out / this.numInputs;
    }

    private float[] calcError(float[] x, float y) {
        assert x.length == this.numInputs : "Input length does not match";

        float[] err = new float[this.numInputs];

        for (int i = 0; i < this.numInputs; i++) {
            err[i] = this.anns[i].run(x[i]) - y;
        }

        return err;
    }

    private float avgError(float[][] x, float[] y) {
        float avg = 0;

        for (int j = 0; j < x.length; j++) {
            float[] err = this.calcError(x[j], y[j]);

            float currentAvg = 0;

            for (int i = 0; i < err.length; i++) {
                currentAvg += err[i];
            }
            avg += currentAvg / err.length;
        }
        return avg / x.length;
    }

    public float fitClassifier(float[][] x, boolean[] y) {
        float[] out = new float[y.length];

        for (int i = 0; i < y.length; i++) {
            if (y[i]) {
                out[i] = 1f;
            } else {
                out[i] = 0f;
            }
        }

        float current_err = this.avgError(x, out);
        int i = 0;

        while (current_err > 0.1f && i < 10_000) {

            for (int j = 0; j < out.length; j++) {
                for (int k = 0; k < this.numInputs; k++) {
                    this.anns[k].backpropigate(x[j][k], out[k]);
                }
            }

            current_err = this.avgError(x, out);

        }

        return current_err;
    }
}
