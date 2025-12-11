package components.ANN;

/**
 * Layered implementations of secondary methods for {@code ANN}.
 */
public abstract class ANNSecondary implements ANN {

    private float mse(float pred, float actual) {
        return (pred - actual) * (pred - actual);
    }

    @Override
    public float fit(float x, float y, float target_err) {
        float current_err = this.mse(this.run(x), y);
        int i = 0;

        while (current_err > target_err && i < 100_000) {
            this.backpropigate(x, y);
            current_err = this.mse(this.run(x), y);
            i++;
        }
        return current_err;
    }

    @Override
    public String toString() {
        float[] weights = this.getW();

        return "ANN(w1=" + weights[0] + " w2=" + weights[1] + ")";
    }

    @Override
    public boolean equals(Object obj) {

        if (this == obj) {
            return true;
        }

        if (obj == null) {
            return false;
        }

        ANN temp = (ANN) obj;
        float[] thisw = this.getW();
        float[] tempw = temp.getW();

        if (thisw[0] == tempw[0] && thisw[1] == tempw[1]) {
            return true;
        }

        return false;
    }

    /*
     * Hash Code is not included because the component is mutable, which would
     * cause changes to the hash code during training. This could cause problems
     * in sorting / hash collections.
     */
}
