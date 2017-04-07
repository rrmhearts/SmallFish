
package cs6850_sentencegraph;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.*;
import edu.uci.ics.jung.algorithms.layout.CircleLayout;
import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.graph.DirectedSparseMultigraph;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import org.apache.commons.collections15.Transformer;

/*
 *
 * @author Ryan McCoppin
 * @date  October 24, 2013
 * @class Foundations of AI
 * 
 * This code works with text files or inline strings. The function to use a 
 *      strings of sentences can be found in the second doNLP function.
 *      Likewise, textfiles can be passed through the command line.
 * 
 * This code is heavily dependent on 
 *      Jung Graphing library and the
 *          * jung-algorithms-2.0.1.jar
 *          * jung-graph-impl-2.0.1.jar
 *          * jung-visualization-2.0.1.jar
 *          * jung-api-2.0.1.jar
 *          * collections-generic-4.01.jar
 *      Stanford NLP Toolkit
 *          * stanford-parser-3.2.0-javadoc.jar
 *          * stanford-parser-3.2.0-models.jar
 * 
 * This code allows the user to use Java style single line comments in the text
 *      file to choose what text is used. There is a text file provided with 
 *      this code that contains example sentences including the class prescribed 
 *      paragraph. Every sentence must end in some punctuation mark (.;!?:) 
 *      followed by a space - as is proper for English.
 */

/*
 *  Relation class
 * 
 *  Creates a relationship between a subject, verb, and direct object strings.
 */
class Relation {
    private String subj;
    private String verb;
    private String obj;
    
    // For unique naming
    public static int counter = 0;
    
    // Constructors
    public Relation(String s, String v, String o) {
        this.subj = s;
        this.verb = v;
        this.obj = o;
        counter++;
    }
    
    public Relation(String s, String v) {
        this.subj = s;
        this.verb = v;
        this.obj = "";
        counter++;
    }
    
    public Relation(String s) {
        this.subj = s;
        this.verb = "";
        this.obj = "";
        counter++;
    }
    
    // Compare two relations by string text
    public boolean compare(Relation n) {
        return 0 == this.subj.compareToIgnoreCase(n.subj) && 
               0 == this.verb.compareToIgnoreCase(n.verb) && 
               0 == this.obj.compareToIgnoreCase(n.obj);
    }
    
    // Getters
    public String getSubject() {
        return this.subj;
    }
    public String getVerb() {
        return this.verb;
    }
    public String getObject() {
        return this.obj;
    }
    
    // Setters
    public void setSubject(String s) {
        this.subj = s;
    }
    public void setVerb(String v) {
        this.subj = v;
    }
    public void setObject(String o) {
        this.subj = o;
    }
    public void nameRelation(String s, String v, String o) {
        this.subj = s;
        this.verb = v;
        this.obj = o;
    }
    
    @Override
    public String toString() { 
        return subj + ' ' + verb + ' ' + obj;
    }

}

public class CS6850_SentenceGraph {
    static String n1,n2,n3,n4;
    
    // Used to prevent duplicates in graph
    static List<String> agents;
    static List<String> actions;
    static List<String> patients;
    
    // Relations for graph
    static List<Relation> relations;
    
    // Our Graph from Jung
    static Graph<String, String> nlpGraph; 
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
        
        // Load the Stanford model
        LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
        
        agents = new ArrayList<String>();
        actions = new ArrayList<String>();
        patients = new ArrayList<String>();
        relations = new ArrayList<Relation>();
        nlpGraph = new DirectedSparseMultigraph<String, String>();
        
        // Apply nlp techniques and for graph
        if (args.length > 0) {
            // For a text file
            doNLP(lp, args[0]);
        } else {
            // For a in code string
            doNLP(lp, nlpGraph);
        }

        System.out.println("The graph g3 = " + nlpGraph.toString());

        // Display graph
        showGraph(nlpGraph);
    } // end main function
    
    /*
     * Perform on a text file provided by user
     */
    public static void doNLP(LexicalizedParser lp, String filename) throws FileNotFoundException, IOException {

        // Read text file into a string
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line, results = "";
        while( ( line = reader.readLine() ) != null)
        {
            // Add lines only not commented out.
            if (!line.contains("//") && !line.startsWith("//")) {
                results += line;
            }
        }
        reader.close();
        
        // Split text into sentences based on punctuation
        String[] sentences = results.split("(?<=[.!?;:] )\\s*"); 
        for (String s : sentences) {
            // If complete Sentence, addSentence to graph
            if (completeSentence(lp, s)) {
                System.out.println("%%% A COMPLETE SENTENCE. %%%");
                System.out.println(s);
                addSentence(lp, s);
                System.out.println("");
            } else {
                System.out.println("");
                System.out.println("%%% NOT A COMPLETE SENTENCE. %%%");
                System.out.println("    " + s);
            }
        }
        removeRepeatedRelations();
        removeRepeatedRelations();
        // Add sentence to graph
        addRelationsToGraph();
    } // end demoDP
    
    /*
     * Perform on the text string provided below
     */
    public static void doNLP(LexicalizedParser lp, Graph<String, String> g) {
        
        //g.addEdge("success", n3, n4); //Comment out later

        // This option shows loading and using an explicit tokenizer
        String text = "Bell, based in Los Angeles, makes and distributes"
                + " electronic, computer and building products. Bell also"
                + " enjoys eating sandwiches and distributing them. Bell is eating a cookie."
                + " Linda also is eating cookies. Linda has given Bell a cookie. Los Angelos has products. "
                + " Linda has been killed. This sentence not complete.";
        
//        text = "Long text – Example description of an image - (Font is decreased to compress area of text.): "
//                + "Image-1, Extracted NL Outcome: "
//                + "Number of objects: 4 objects recognized in the input image. "
//                + "Type of objects: A yellow car, a brown airplane, a black helicopter and a silver wrench-tool "
//                + "Location of the objects: The airplane is on the upper left part of the image. Its position is “nose-SE-side”. "
//                + "The helicopter is on the upper right part of the image. Its position is “nose-NE-side”. "
//                + "The wrench-tool is on the upper center part of the image. Its position is “open-edge-NE-side”. "
//                + "The car is on the lower center part of the image. Its position is “nose-W-side”."
//                + "Associations of the objects: The car is 2.1 the length of the airplane, 2.3 the length of the helicopter, 1.3 the length of the wrench-tool; "
//                + "The car is the biggest in length object; The airplane is the second in length object; The helicopter and the wrench-tool have similar length; "
//                + "Image-2, Extracted NL Outcome: Number of objects: 4 objects recognized in the input image. "
//                + "Type of objects: A yellow car, a brown airplane, a black handle hummer and a silver wrench-tool "
//                + "Location of the objects: The airplane is on the lower central part of the image. Its position is “nose-NE-side”. "
//                + "The hummer is on the middle-diagonal of the image. Its position is “back-side-top-NE” The wrench-tool is on the right-down part of the image. "
//                + "Its position is “open-edge-top-NW-side”. The car is on the upper left part of the image. Its position is “right-side-nose-NE”. "
//                + "Associations of the objects: The hummer is the longest object with 2.1 times the length of the wrench-tool, with 3 times the length of the airplane, "
//                + "with 1.3 the length of the car; The car is 1.2 the length of the wrench-tool, 1.8 of the airplane; The car is second in length object; The airplane is the smallest object;";
        
        String[] sentences = text.split("(?<=[.!?;:] )\\s*"); // need to NOT break on numbers

        for (String s : sentences) {
            if (completeSentence(lp, s)) {
                System.out.println("%%% A COMPLETE SENTENCE. %%%");
                System.out.println(s);
                addSentence(lp, s);
                System.out.println("");
            } else {
                System.out.println("");
                System.out.println("%%% NOT A COMPLETE SENTENCE. %%%");
                System.out.println("    " + s);
            }
        }
        removeRepeatedRelations();
        removeRepeatedRelations();
        // Add sentence to graph
        addRelationsToGraph();
        
    } //end demoAPI
    
    /*
     * Add sentence to the relation set
     */
    public static void addSentence(LexicalizedParser lp, String text) {
        TokenizerFactory<CoreLabel> tokenizerFactory =
          PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
        List<CoreLabel> rawWords =
          tokenizerFactory.getTokenizer(new StringReader(text)).tokenize();
        Tree parse = lp.apply(rawWords);
      
        TreebankLanguagePack tlp = new PennTreebankLanguagePack();
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        
        System.out.println(" % % % Typed dependencies % % %");
        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
        System.out.println(tdl);
        
        for (TypedDependency rel : tdl) {
            // NEW
            addRelation(rel, tdl);
        }

    }
    
    /*
     * Remove relations that occur more than once
     */
    public static void removeRepeatedRelations() {
        int Length = relations.size();
        for (int i = 0; i < Length; i++) {
            Relation curr = relations.get(i);
            for (Relation j : relations) {
                if (!curr.equals(j) && (curr.compare(j) ||
                         curr.getSubject().compareToIgnoreCase(j.getSubject())==0 && 
                            curr.getVerb().compareToIgnoreCase(j.getVerb())==0 &&
                         j.getObject().isEmpty() ) ) {
//                    System.out.println("Removed: " + j.toString());
                    relations.remove(j);
                    Length--;
                    break;
                } else if (!curr.equals(j) && (curr.compare(j) ||
                         curr.getObject().compareToIgnoreCase(j.getObject())==0 && 
                            curr.getVerb().compareToIgnoreCase(j.getVerb())==0 &&
                         j.getSubject().isEmpty() ) ) {
//                    System.out.println("Removed: " + j.toString());
                    relations.remove(j);
                    Length--;
                    break;
                } //end if
            } // end for j relations
        }
    } //end removeRepeatedRelations function
    
    /*
     * Add relations to graph
     */
    public static void addRelationsToGraph() {
        int count = 0;
        String sub, act, obj;
        for (Relation rel : relations) {
            //System.out.println("relations: " + rel.toString());
            
            // Add relations to node lists to ensure unique nodes. (not needed, but didn't remove)
            if (!agents.contains(rel.getSubject()) && rel.getSubject().compareTo("")!=0) {
                agents.add(rel.getSubject());
                nlpGraph.addVertex(agents.get(agents.indexOf(rel.getSubject())));//rel.getSubject());
            }
            if (!actions.contains(rel.getVerb()) && rel.getVerb().compareTo("")!=0) {
                actions.add(rel.getVerb());
                nlpGraph.addVertex(actions.get(actions.indexOf(rel.getVerb())));//rel.getVerb());
            }
            if (!patients.contains(rel.getObject()) && rel.getObject().compareTo("")!=0) {
                patients.add(rel.getObject());
                nlpGraph.addVertex(patients.get(patients.indexOf(rel.getObject())));//rel.getObject());
            }
            
            // Add relation to graph from it's node, if subject-verb relation
            if (rel.getSubject().compareToIgnoreCase("")!=0 && rel.getVerb().compareToIgnoreCase("")!=0)
            {
                sub = agents.get(agents.indexOf(rel.getSubject()));
                act = actions.get(actions.indexOf(rel.getVerb()));
                nlpGraph.addEdge(count + "s-v" + rel.counter, sub, act);
            }
            // Add relation to graph from it's node, if verb-object relation
            if (rel.getObject().compareToIgnoreCase("")!=0 && rel.getVerb().compareToIgnoreCase("")!=0)
            {
                act = actions.get(actions.indexOf(rel.getVerb()));
                obj = patients.get(patients.indexOf(rel.getObject()));
                nlpGraph.addEdge(count + "v-o" + rel.counter, act, obj);
            }
            // Must have a unique name
            count++;
        } // end for relations
    } // end addRelationsToGraph
    
    /*
     * True if the word is in this relation found in tdl
     */
    public static boolean thisISIN(String word, String inThisRelation, List<TypedDependency> tdl) {
        for (TypedDependency rel : tdl) {
            //rel.gov().toString().replaceAll("-[^-]*", "");
            if (rel.reln().toString().compareToIgnoreCase(inThisRelation)==0) {
                // Word must match gov term or dep term
                if (rel.gov().toString().replaceAll("-*\\d*$", "").compareToIgnoreCase(word)==0 || 
                        rel.dep().toString().replaceAll("-*\\d*$", "").compareToIgnoreCase(word)==0 ) {
                    return true;
                    
                }
            }
        }
        return false;
    }
    
    /*
     * Returns the opposite term in relation from word. 
     *          If word is dep, return gov
     *          IF word is gov, return dep
     */
    public static String grabPartner(String word, String inThisRelation, List<TypedDependency> tdl) {
        for (TypedDependency rel : tdl) {
            if (rel.reln().toString().compareToIgnoreCase(inThisRelation)==0) {
                if (rel.gov().toString().replaceAll("-*\\d*$", "").compareToIgnoreCase(word)==0) {
                    return rel.dep().toString().replaceAll("-*\\d*$", "");
                }
                if (rel.dep().toString().replaceAll("-*\\d*$", "").compareToIgnoreCase(word)==0 ) {
                    return rel.gov().toString().replaceAll("-*\\d*$", "");                    
                }
            }
        }
        
        return "";
    }
    
    /*
     * Is text a complete sentence. A sentence must have a subject and verb.
     */
    public static boolean completeSentence(LexicalizedParser lp, String text) {
        TokenizerFactory<CoreLabel> tokenizerFactory =
          PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
        List<CoreLabel> rawWords =
          tokenizerFactory.getTokenizer(new StringReader(text)).tokenize();
        Tree parse = lp.apply(rawWords);
      
        TreebankLanguagePack tlp = new PennTreebankLanguagePack();
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        
        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
        
        boolean subv = false, vobj = false;
        // Sentence must have a nsubj term or nsubjpass term 
        for (TypedDependency rel : tdl) {
            if ( ( rel.reln().toString().compareToIgnoreCase("nsubj")==0 ||
                   rel.reln().toString().compareToIgnoreCase("nsubjpass")==0 ||
                   rel.reln().toString().compareToIgnoreCase("xsubj")==0) ) {
                subv = true;
            } else if ( ( rel.reln().toString().compareToIgnoreCase("cop")==0 ||
                   rel.reln().toString().compareToIgnoreCase("aux")==0 ||
                   rel.reln().toString().compareToIgnoreCase("dobj")==0 ||
                   rel.reln().toString().compareToIgnoreCase("xcomp")==0 ||
                   rel.reln().toString().compareToIgnoreCase("ccomp")==0 ||
                   rel.reln().toString().compareToIgnoreCase("auxpass")==0) ) {
                vobj = true;
            } 
        }
        if (subv && vobj) {
            return true;
        }
              
        return false;
    }
    
    /*
     * Add relation rel to relations list
     *      Heavily dependent on "Stanford Dependencies"
     *      All the "AI" work is done here through if statements
     */
    public static void addRelation(TypedDependency rel, List<TypedDependency> tdl) {
        String relationType = rel.reln().toString();
        
        // First term: nsubj(gov, dep)
        String gov = rel.gov().toString().replaceAll("-*\\d*$", ""); //"-[^a-z]*$", "");
        // Second term: nsubj(gov, dep)
        String dep = rel.dep().toString().replaceAll("-*\\d*$", "");
        String subject = "", object = "", verb = "";
        if (relationType.compareToIgnoreCase("nsubj") == 0) {
            // A nominal subject is a noun phrase which is the syntactic subject 
            // of a clause. The governor of this relation might not always be a 
            // verb: when the verb is a copular verb, the root of the clause is 
            // the complement of the copular verb, which can be an adjective or noun.
            
            // If the subject is a nn
            if (thisISIN(dep, "nn", tdl)) {
                subject = grabPartner(dep, "nn", tdl) + " " + dep;
            } else {
                subject = dep;
            }
            // If the gov is a noun
            if (thisISIN(gov, "cop", tdl)) {
                object = gov;
                verb = grabPartner(gov, "cop", tdl);
                relations.add((new Relation("", verb, object)));
            } else if (thisISIN(gov, "aux", tdl)) {
             //An auxiliary of a clause is a non-main verb of the clause, e.g. 
             // modal auxiliary, “be” and “have” in a composed tense.
                verb = grabPartner(gov, "aux", tdl) + " " + gov;
            } else {
                verb = gov;
            }
                relations.add(new Relation(subject, verb));
        } else if (relationType.compareToIgnoreCase("nsubjpass") == 0) {
            // A passive nominal subject is a noun phrase which is the syntactic 
            //   subject of a passive clause.
            
            // If the object is a nn
            if (thisISIN(dep, "nn", tdl)) {
                object = grabPartner(dep, "nn", tdl) + " " + dep;
            } else {
                object = dep;
            }
            
            if (thisISIN(gov, "aux", tdl)) {
//                object = gov;
                verb = grabPartner(gov, "aux", tdl) + " " + gov;
            } else {
                verb = gov;
            }            
                relations.add(new Relation("", verb, object));
        } else if (relationType.compareToIgnoreCase("dobj") == 0) {
            // The direct object of a VP is the noun phrase which is the 
            // (accusative) object of the verb.
            if (thisISIN(gov, "ccomp", tdl)) {
                verb = grabPartner(gov, "ccomp", tdl);
                object = gov + " " + dep;
            } else if (thisISIN(gov, "xcomp", tdl)) {
                verb = grabPartner(gov, "xcomp", tdl);
                object = gov + " " + dep;
            } else if (thisISIN(gov, "aux", tdl)) {
                verb = grabPartner(gov, "aux", tdl) + " " + gov;
                object = dep;
            } else if (thisISIN(gov, "auxpass", tdl)) {
                verb = grabPartner(gov, "auxpass", tdl) + " " + gov;
                object = dep;
            }else {
                verb = gov;
                object = dep;
                // If verb is in conjunction with another verb, have second verb also point to direct object
                if (thisISIN(gov, "conj_and", tdl) && verb.compareToIgnoreCase(gov)==0) {
                    relations.add(new Relation("", grabPartner(gov, "conj_and", tdl), object));
                }
            }
            relations.add(new Relation("", verb, object));
        } else if (relationType.compareToIgnoreCase("agent") == 0) {
            // An agent is the complement of a passive verb which is introduced 
            // by the preposition “by” and does the action.
            subject = dep;
            verb = gov;
            relations.add(new Relation(subject, verb));
        } else if (relationType.compareToIgnoreCase("attr") == 0) {
            // An attributive is a complement of a copular verb such as “to be”,
            // “to seem”, “to appear”. Currently, the converter only recognizes WHNP complements
            subject = dep;
            verb = gov;
            relations.add(new Relation(subject, verb));
        } else if (relationType.compareToIgnoreCase("cop") == 0) {
            // A copula is the relation between the complement of a copular 
            // verb and the copular verb.
            verb = dep; 
            object = gov;
            relations.add(new Relation("", verb, object));
        } else if (relationType.compareToIgnoreCase("ccomp") == 0) {
            //A clausal complement of a verb or adjective is a dependent clause 
            // with an internal subject which functions like an object of the 
            // verb, or adjective. C "enjoys, distributing"
            verb = gov;
            if (thisISIN(dep, "dobj", tdl)) {
                object = dep + " " + grabPartner(dep, "dobj", tdl);
            } else {
                object = dep;
            }

            relations.add(new Relation("", verb, object));
        } else if (relationType.compareToIgnoreCase("expl") == 0) {
            // This relation captures an existential “there”. The main verb 
            // of the clause is the governor.
            subject = dep; 
            verb = gov;
            relations.add(new Relation(subject, verb));
        }
    } // end addRelations function
    
    /*
     * Show graph function. Assumes Graph<String, String>
     */
    public static void showGraph(Graph g)
    {
        // The Layout<V, E> is parameterized by the vertex and edge types
        Layout<String, String> layout = new CircleLayout(g);
        VisualizationViewer<String,String> vv = 
            new VisualizationViewer<String,String>(layout);
        
        int Size = 600 + 20*nlpGraph.getVertexCount();
        Size = Math.min(Size, 1000);
        vv.setPreferredSize(new Dimension(Size,Size)); //Sets the viewing area size
        
        // Setup up a new vertex to paint transformer...
        Transformer<String,Paint> vertexPaint = new Transformer<String,Paint>() {
            public Paint transform(String i) {
                return Color.GREEN;
            }
        };
        Transformer<String,Shape> vertexSize = new Transformer<String,Shape>(){
            public Shape transform(String i) {
                final int Size = 10 + (90/nlpGraph.getVertexCount());
                Ellipse2D circle = new Ellipse2D.Double(-5, -5, Size, Size);
                // in this case, the vertex is twice as large
                return AffineTransform.getScaleInstance(2, 2).createTransformedShape(circle);
            }
        };
        
        vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
        vv.getRenderContext().setVertexShapeTransformer(vertexSize);
        
        vv.getRenderContext().setVertexLabelTransformer(new ToStringLabeller());
        vv.getRenderContext().setLabelOffset(0);

        JFrame frame = new JFrame("Simple Graph View");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(vv); 
        
        frame.pack();
        frame.setVisible(true); 
    } // end showGraph
} // End class