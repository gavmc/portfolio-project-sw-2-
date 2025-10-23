public interface ANN extends ANNKernel {

    /*
     * Fits the input value {@code x} to the target output value {@code y}
     *
     * @param x The input value
     * 
     * @param y The target output value
     * 
     * @updates this
     */
    float fit(float x, float y);
}
