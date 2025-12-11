import components.ANN.ANN;
import components.ANN.ANN1L;
import components.random.Random;
import components.random.Random1L;

public class PasswordHashExample {

    private ANN ann = new ANN1L();
    private Random rand = new Random1L();

    private float err = 0.00001f;

    public PasswordHashExample() {
        this.ann.fit(1f, (float) this.rand.nextDouble(), 0.001f);
    }

    public PasswordHashExample(float seed) {
        this.ann.fit(1f, seed, 0.0001f);
    }

    public float getHash(float input) {
        return this.ann.run(input);
    }

    public void scrambleHash() {
        this.ann.fit(1f, (float) this.rand.nextDouble(), 0.001f);
    }

    public void scrambleHash(float seed) {
        this.ann.fit(1f, seed, 0.001f);
    }

    public boolean checkEquals(float input, float hashed) {
        float val = this.ann.run(input);
        return Math.abs(val - hashed) < this.err;
    }
}
