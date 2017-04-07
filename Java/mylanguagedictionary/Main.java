package mylanguagedictionary;

/**
 * @author Goshikku
 */

public class Main {

    public static void main(String[] args) {
        Dictionary dict = new Dictionary("mylang.txt");
        
        dict.writeDictionary("asa", "I");
        dict.writeDictionary("ora", "or");
    }

}
