package components.ANN;

/**
 * {@code ANNKernel} with an added secondary method (fit)
 */
public interface ANN extends ANNKernel {

    /*
     * Fits the input value {@code x} to the target output value {@code y}
     *
     * @param x The input value
     *
     * @param y The target output value
     *
     * @updates this
     *
     * @ensures fit = (this.run(x), y) * (this.run(x), y)
     */
    float fit(float x, float y, float target_err);
}
