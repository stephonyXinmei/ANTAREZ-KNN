import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.StringUtils;

import java.io.*;
import java.net.URI;
import java.util.*;

public class KnnMapper extends Mapper<Object, Text, NullWritable, DistanceQuery> {

    private static final int VEC_DIMENSION = 300;

    private static int K;

    private static DistanceQuery distanceAndQuery = new DistanceQuery();
    private static TreeMap<Double, String> knnMap = new TreeMap<Double, String>(new Comparator<Double>() {
        @Override
        public int compare(Double o1, Double o2) {
            return o2.compareTo(o1);
        }
    });
    private static List<Vector<Double>> traceFeatureVecs = new ArrayList<Vector<Double>>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        K = Integer.valueOf(context.getConfiguration().get(Constants.PARAM_K));
        URI[] cacheFiles = context.getCacheFiles();
        if (cacheFiles != null && cacheFiles.length > 0) {
            for (URI cacheFile: cacheFiles) {
                String filename = cacheFile.getPath().toString();
                traceFeatureVecs.addAll(readFileToVectorList(filename));
            }
        }
    }

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] cols = line.split("\t");

        String query = cols[0];
        Vector<Double> featureVec = parseStringToVector(cols[1]);

        for (Vector<Double> traceFeatureVec : traceFeatureVecs) {
            Double dist = FeatureSimilarity.getCosineSimilarity(traceFeatureVec, featureVec);
            if (dist > 0.0) {
                knnMap.put(dist, query);
                if (knnMap.size() > K) {
                    knnMap.remove(knnMap.lastKey());
                }
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Map.Entry<Double, String> entry : knnMap.entrySet()) {
            Double knnDist = entry.getKey();
            String knnQuery = entry.getValue();

            distanceAndQuery.set(knnDist, knnQuery);
            context.write(NullWritable.get(), distanceAndQuery);
        }
    }

    private List<Vector<Double>> readFileToVectorList(String filename) {
        List<Vector<Double>> retList = new ArrayList<Vector<Double>>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line = null;
            while ((line = reader.readLine()) != null) {
                retList.add(parseStringToVector(line));
            }
        } catch (IOException e) {
            System.err.println("Caught exception while parsing the cached file '"
                    + StringUtils.stringifyException(e));
        }
        return retList;
    }

    private Vector<Double> parseStringToVector(String line) {
        Vector<Double> retVec = new Vector<Double>(VEC_DIMENSION);
        String[] cols = line.split(",");
        for (String x : cols) {
            retVec.addElement(Double.valueOf(x));
        }
        return retVec;
    }
}
