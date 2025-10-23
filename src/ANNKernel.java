import components.standard.Standard;

public interface ANNKernel extends Standard<ANN> {

    /*
     * Runs though the single node network with {@code x} as the input
     *
     * @param x the input to the network
     */
    float run(float x);

    /*
     * Does a single backpropigation loop
     *
     * @param x The input value
     *
     * @param y The target output value
     *
     * @updates this
     */
    void backpropigate(float x, float y);

}
