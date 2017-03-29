import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

public class KnnReducer extends Reducer<NullWritable, DistanceQuery, NullWritable, Text> {

    private static int K;

    private static TreeMap<Double, String> knnMap = new TreeMap<Double, String>(new Comparator<Double>() {
        @Override
        public int compare(Double o1, Double o2) {
            return o2.compareTo(o1);
        }
    });

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        K = Integer.valueOf(context.getConfiguration().get(Constants.PARAM_K));
    }

    @Override
    protected void reduce(NullWritable key, Iterable<DistanceQuery> values, Context context) throws IOException, InterruptedException {
        for (DistanceQuery val : values) {
            Double dist = val.getDistance();
            String query = val.getQuery();

            knnMap.put(dist, query);
            if (knnMap.size() > K) {
                knnMap.remove(knnMap.lastKey());
            }
        }

        for (Map.Entry<Double, String> entry : knnMap.entrySet()) {
            context.write(NullWritable.get(), new Text(String.valueOf(entry.getKey()) + "\t" + entry.getValue()));
        }
    }


}
