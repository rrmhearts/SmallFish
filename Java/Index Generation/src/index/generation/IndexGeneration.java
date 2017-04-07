/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package index.generation;






import java.io.*;
import java.util.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class IndexGeneration extends Configured implements Tool {
   // Develop a mapper and reducer program to extract information
   //    from each mp3 file and produce and index of the songs
   //    based on the artist name.
   public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {

     static enum Counters { INPUT_WORDS }

     private final static IntWritable one = new IntWritable(1);
     private Text word = new Text();

     private boolean caseSensitive = true;
     private long numRecords = 0;
     private String inputFile;

     private void getArtist(File file) {
		String artist;// = "None";
		try {
			FileInputStream in = new FileInputStream(file);
			int fileSize =  (int)file.length();
			in.skip(fileSize - 128);
			byte[] tagBytes = new byte[128];
			in.read(tagBytes, 0, tagBytes.length);
			String id3 = new String(tagBytes);
			String tag = id3.substring(0,3);
			if (tag.equals("TAG")) {
				artist = id3.substring(33, 62).trim();
				//patternsToSkip.add(artist);
			}
		} catch (IOException e) {
			System.err.println("Caught exception while parsing the cached file '" + file.getName() + "' : " + StringUtils.stringifyException(e));
		}
     } // end getArtist
     
     private void getTitle(File file) {
		String title;// = "None";
		try {
			FileInputStream in = new FileInputStream(file);
			int fileSize =  (int)file.length();
			in.skip(fileSize - 128);
			byte[] tagBytes = new byte[128];
			in.read(tagBytes, 0, tagBytes.length);
			String id3 = new String(tagBytes);
			String tag = id3.substring(0,3);
			if (tag.equals("TAG")) {
				title = id3.substring(3, 32).trim();
				//patternsToSkip.add(artist);
			}
		} catch (IOException e) {
			System.err.println("Caught exception while parsing the cached file '" + file.getName() + "' : " + StringUtils.stringifyException(e));
		}
     } // end getTitle
     

     public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
        String id3 = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
        String artist = "", title = "";
        String tag = id3.substring(0,3);
        if (tag.equals("TAG")) {
            artist = id3.substring(33, 62).trim();
            title = id3.substring(3, 32);
        }
        output.collect(new Text(artist), new Text(title));
        reporter.incrCounter(Counters.INPUT_WORDS, 1);

        if ((++numRecords % 100) == 0) {
             reporter.setStatus("Finished processing " + numRecords + " records " + "from the input file: " + inputFile);
        }
     } // end map
   } // end class

   public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
     public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
       String sum = "";
       while (values.hasNext()) {
         sum += values.next().toString() + " ";
       }
       output.collect(key, new Text(sum));
     }
   }

   public int run(String[] args) throws Exception {
     JobConf conf = new JobConf(getConf(), IndexGeneration.class);
     conf.setJobName("indexgen");

     conf.setOutputKeyClass(Text.class);
     conf.setOutputValueClass(IntWritable.class);

     conf.setMapperClass(Map.class);
     conf.setCombinerClass(Reduce.class);
     conf.setReducerClass(Reduce.class);

     conf.setInputFormat(TextInputFormat.class);
     conf.setOutputFormat(TextOutputFormat.class);

     conf.setJarByClass(IndexGeneration.class);

     List<String> other_args = new ArrayList<String>();
     for (int i=0; i < args.length; ++i) {
       if ("-skip".equals(args[i])) {
         DistributedCache.addCacheFile(new Path(args[++i]).toUri(), conf);
         conf.setBoolean("indexgen.skip.patterns", true);
       } else {
         other_args.add(args[i]);
       }
     }

     FileInputFormat.setInputPaths(conf, new Path(other_args.get(0)));
     FileOutputFormat.setOutputPath(conf, new Path(other_args.get(1)));

     JobClient.runJob(conf);
     return 0;
   }

   public static void main(String[] args) throws Exception {
     int res = ToolRunner.run(new Configuration(), new IndexGeneration(), args);
     System.exit(res);
   }
}


