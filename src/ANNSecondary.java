public abstract class ANNSecondary implements ANN {

    private float mse(float pred, float actual) {
        return (pred - actual) * (pred - actual);
    }

    @Override
    public float fit(float x, float y, float target_err) {
        float current_err = this.mse(this.run(x), y);
        int i = 0;

        while (current_err > target_err && i < 10_000) {
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
        ANN temp = (ANN) obj;
        if (this.getW() == temp.getW()) {
            return true;
        }
        return false;
    }
}
