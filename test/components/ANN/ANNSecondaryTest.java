package components.ANN;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

public class ANNSecondaryTest {

    @Test
    public void fitTest1() {
        ANN ann = new ANN1L(0.001f);

        float input = 1f;
        float expectedOutput = 2f;

        float targetErr = 0.0001f;

        float err = ann.fit(input, expectedOutput, targetErr);

        assertEquals(true, err < targetErr);
        assertEquals(expectedOutput, ann.run(input), Math.sqrt(targetErr));
    }

    @Test
    public void fitTest2() {
        ANN ann = new ANN1L(0.001f);

        float input = 12f;
        float expectedOutput = -2f;

        float targetErr = 0.0001f;

        float err = ann.fit(input, expectedOutput, targetErr);

        assertEquals(true, err < targetErr);
        assertEquals(expectedOutput, ann.run(input), Math.sqrt(targetErr));
    }

    @Test
    public void fitTest3() {
        ANN ann = new ANN1L(0.001f);

        float input = 0.1f;
        float expectedOutput = 100f;

        float targetErr = 0.0001f;

        float err = ann.fit(input, expectedOutput, targetErr);

        assertEquals(true, err < targetErr);
        assertEquals(expectedOutput, ann.run(input), Math.sqrt(targetErr));
    }

    @Test
    public void sameEqualsTest() {
        ANN ann = new ANN1L();

        assertEquals(ann, ann);
    }

    @Test
    public void diffEqualsTest() {
        ANN ann1 = new ANN1L();
        ANN ann2 = new ANN1L();

        assertEquals(ann1, ann2);
    }

    @Test
    public void notEqualsTest() {
        ANN ann1 = new ANN1L();
        ANN ann2 = new ANN1L();

        ann1.backpropigate(1f, 2f);

        assertNotEquals(ann1, ann2);
    }

    @Test
    public void stringTest1() {
        ANN ann = new ANN1L();

        float[] weights = ann.getW();

        String expected = "ANN(w1=" + weights[0] + " w2=" + weights[1] + ")";

        assertEquals(expected, ann.toString());
    }

    @Test
    public void stringTest2() {
        ANN ann = new ANN1L();

        ann.fit(1f, 2f, 0.0001f);

        float[] weights = ann.getW();

        String expected = "ANN(w1=" + weights[0] + " w2=" + weights[1] + ")";

        assertEquals(expected, ann.toString());
    }
}
