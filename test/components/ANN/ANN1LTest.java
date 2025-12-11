package components.ANN;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

public class ANN1LTest {

    @Test
    public void defaultConstructorTest() {
        ANN ann = new ANN1L();

        float[] weights = ann.getW();
        float delta = 0.0001f;

        assertEquals(1.0f, weights[0], delta);
        assertEquals(1.0f, weights[1], delta);
    }

    @Test
    public void setRunTest1() {
        ANN ann = new ANN1L();

        float num = ann.run(1);
        float delta = 0.0001f;

        // w1 = 1f, w2 = 1f
        // l1 = sigmoid(1 * w1) = sigmoid(1) = 0.73105857863
        // l2 = 0.73105857863 * w1 = 0.73105857863
        // num = l1 = 0.73105857863

        assertEquals(0.73105857863f, num, delta);
    }

    @Test
    public void setRunTest2() {
        ANN ann = new ANN1L();

        float num = ann.run(2.5f);
        float delta = 0.0001f;

        // w1 = 1f, w2 = 1f
        // l1 = sigmoid(2.5 * w1) = sigmoid(2.5) = 0.924141819979
        // l2 = 0.924141819979 * w1 = 0.924141819979
        // num = l1 = 0.924141819979

        assertEquals(0.924141819979f, num, delta);
    }

    @Test
    public void setRunTest3() {
        ANN ann = new ANN1L();

        float num = ann.run(-3);
        float delta = 0.0001f;

        // w1 = 1f, w2 = 1f
        // l1 = sigmoid(-3 * w1) = sigmoid(-3) = 0.0474258731776
        // l2 = 0.0474258731776 * w1 = 0.0474258731776
        // num = l1 = 0.0474258731776

        assertEquals(0.0474258731776f, num, delta);
    }

    @Test
    public void weightBackpropTest() {
        ANN ann = new ANN1L(0.01f);

        float input = 2.5f;
        float output = 0.5f;

        float delta = 0.00001f;

        float[] preWeights = ann.getW();

        ann.backpropigate(input, output);

        float[] postWeights = ann.getW();

        assertNotEquals(preWeights[0], postWeights[0], delta);
        assertNotEquals(preWeights[1], postWeights[1], delta);
    }

    @Test
    public void lrConstructorTest() {
        ANN ann1 = new ANN1L(0.05f);
        ANN ann2 = new ANN1L(0.001f);

        float input = 2.5f;
        float output = 0.5f;

        float[] preWeights1 = ann1.getW();
        float[] preWeights2 = ann2.getW();

        ann1.backpropigate(input, output);
        ann2.backpropigate(input, output);

        float[] postWeights1 = ann1.getW();
        float[] postWeights2 = ann2.getW();

        float[] diff1 = new float[2];
        float[] diff2 = new float[2];

        diff1[0] = Math.abs(postWeights1[0] - preWeights1[0]);
        diff1[1] = Math.abs(postWeights1[1] - preWeights1[1]);

        diff2[0] = Math.abs(postWeights2[0] - preWeights2[0]);
        diff2[1] = Math.abs(postWeights2[1] - preWeights2[1]);

        assertEquals(true, diff1[0] > diff2[0]);
        assertEquals(true, diff1[1] > diff2[1]);

    }

    @Test
    public void clearTest() {
        ANN ann = new ANN1L();

        ann.fit(1f, 2f, 0.001f);

        assertNotEquals(ann, new ANN1L());

        ann.clear();

        assertEquals(ann, new ANN1L());
    }

    @Test
    public void newInstanceTest() {
        ANN ann1 = new ANN1L();

        ann1.fit(1f, 2f, 0.001f);

        ANN ann2 = ann1.newInstance();

        float[] weights = ann2.getW();

        assertEquals(1f, weights[0], 0.0001f);
        assertEquals(1f, weights[1], 0.0001f);
    }

    @Test
    public void transferTest() {
        ANN ann1 = new ANN1L();
        ANN ann2 = new ANN1L();

        ann1.fit(1f, 2f, 0.001f);

        float out = ann1.run(1f);
        ann2.transferFrom(ann1);

        assertEquals(out, ann2.run(1f), 0.0001f);
        assertEquals(ann1, new ANN1L());
    }
}
