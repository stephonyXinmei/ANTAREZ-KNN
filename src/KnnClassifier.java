import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.net.URI;

public class KnnClassifier {

    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.err.println("Usage: KnnPattern <in> <out> <aid-vectors dir> <aid> <K>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.set(Constants.PARAM_AID, args[3]);
        conf.set(Constants.PARAM_K, args[4]);

        Job job = Job.getInstance(conf, "Find K-Nearest Neighbour");

        job.setMapperClass(KnnMapper.class);
        job.setReducerClass(KnnReducer.class);
        job.setNumReduceTasks(1);

        URI[] toCacheFileURIs = getSubFileURIs(conf, args[2] + "/aid=" + args[3], new PathFilter() {
            @Override
            public boolean accept(Path path) {
                return path.toString().contains("part-");
            }
        });

        for (URI uri : toCacheFileURIs) {
            System.out.println("[ruic-log]: " + uri.getPath().toString() );
        }

        job.setCacheFiles(toCacheFileURIs);

        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(DistanceQuery.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    private static URI[] getSubFileURIs(Configuration conf, String hdfsDir, PathFilter filter) throws Exception {
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] fileStatuses = fs.listStatus(new Path(hdfsDir), filter);

        int fileNum = fileStatuses.length;
        URI[] retURIs = new URI[fileNum];

        for (int i = 0; i < fileNum; ++i) {
            retURIs[i] = new URI(fileStatuses[i].getPath().toString());
        }

        return retURIs;
    }
}
