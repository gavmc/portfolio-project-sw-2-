
/**
 * {@code ANN} represented as a {@link float} with implementations of primary
 * methods.
 *
 * @convention $this.lr >= 0
 * @correspondence this = $this.w1. $this.w2, $this.lr
 */
public class ANN1L extends ANNSecondary {

    /**
     * representation of {@code this}.
     */
    private float w1;
    private float w2;
    private float lr;

    /**
     * Create an initial representation
     */
    private void createNewRep() {
        this.w1 = 1f;
        this.w2 = 1f;
        this.lr = 0.01f;
    }

    /**
     * constructor with no argument
     */
    public ANN1L() {
        this.createNewRep();
    }

    /**
     * constructor with learning rate paramater
     *
     * @param lr
     *            {@code float} to initalize lr from
     */
    public ANN1L(float lr) {
        assert lr > 0 : "Violation of lr > 0";

        this.createNewRep();
        this.lr = lr;
    }

    @Override
    public final ANN newInstance() {
        try {
            return this.getClass().getConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(
                    "Cannot construct object of type " + this.getClass());
        }
    }

    @Override
    public final void clear() {
        this.createNewRep();
    }

    @Override
    public final void transferFrom(ANN source) {
        assert source != null : "Violation of: source is not null";
        assert source != this : "Violation of: source is not this";
        assert source instanceof ANN1L : ""
                + "Violation of: source is of dynamic type ANN1L";

        ANN1L localSource = (ANN1L) source;
        this.w1 = localSource.w1;
        this.w2 = localSource.w2;
        this.lr = localSource.lr;
        localSource.createNewRep();
    }

    //----------------------------------------------------

    /**
     * Signmoid helper function
     * 
     * @param x
     *            {@code float} to input into the sigmoid function
     * @return {@code float} output of the sigmoid function
     */
    private final float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp((double) -x)));
    }

    /**
     * Signmoid derivative helper function
     * 
     * @param x
     *            {@code float} to input into the sigmoid derivative function
     * @return {@code float} output of the sigmoid derivative function
     */
    private final float sigmoid_deriv(float x) {
        return x * (1 - x);
    }

    /**
     * mean squared error helper function
     * 
     * @param pred
     *            prediction by ANN
     * @param actual
     *            Label from data
     * @return Mean squared error between actual and pred
     */
    private final float mse(float pred, float actual) {
        return (pred - actual) * (pred - actual);
    }

    /**
     * Run with more infmation for backpropigation
     * 
     * @param x
     *            Input value
     * @return hidden layer values + output
     */
    private final float[] runInfo(float x) {
        float[] info = new float[2];
        info[0] = this.sigmoid(x * this.w1);
        info[1] = this.sigmoid(info[0] * this.w2);
        return info;
    }

    //--------------------------------------------------

    @Override
    public final float[] getW() {
        float[] weights = new float[2];
        weights[0] = this.w1;
        weights[1] = this.w2;
        return weights;
    }

    @Override
    public final void backpropigate(float x, float y) {
        float[] info = this.runInfo(x);

        float err_2 = this.mse(info[1], y);
        float delta_2 = err_2 * this.sigmoid_deriv(info[1]);

        float err_1 = delta_2 * this.w2;
        float delta_1 = err_1 * this.sigmoid_deriv(info[0]);

        this.w2 += info[0] * delta_2 * this.lr;
        this.w1 += x * delta_1 * this.lr;
    }

    @Override
    public final float run(float x) {
        x = this.sigmoid(x * this.w1);
        x = this.sigmoid(x * this.w2);
        return x;
    }
}
