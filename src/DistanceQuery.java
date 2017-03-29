import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by xm on 27/03/2017.
 */
public class DistanceQuery implements WritableComparable<DistanceQuery> {

    private Double distance = 0.0;
    private String query = null;

    public void set(Double lhs, String rhs) {
        distance = lhs;
        query = rhs;
    }

    public Double getDistance() {
        return distance;
    }

    public String getQuery() {
        return query;
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        distance = dataInput.readDouble();
        query = dataInput.readUTF();
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeDouble(distance);
        dataOutput.writeUTF(query);
    }

    @Override
    public int compareTo(DistanceQuery o) {
        return this.query.compareTo(o.query);
    }
}
