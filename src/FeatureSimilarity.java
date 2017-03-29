import java.util.Vector;

/**
 * Created by xm on 27/03/2017.
 */
public class FeatureSimilarity {

    public static Double getCosineSimilarity(Vector<Double> featureVec1, Vector<Double> featureVec2) {
        double product = 0.0;
        double norm1 = 0.0, norm2 = 0.0;

        int dim = featureVec1.size();
        for (int i = 0; i < dim; ++i) {
            product += featureVec1.get(i) * featureVec2.get(i);
            norm1 += Math.pow(featureVec1.get(i), 2);
            norm2 += Math.pow(featureVec2.get(i), 2);
        }

        return product / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

}
