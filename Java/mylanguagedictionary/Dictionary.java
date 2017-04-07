/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mylanguagedictionary;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

/**
 *
 * @author Goshikku
 */
public class Dictionary {

    File fileHandle;
    FileWriter writer;
    public Dictionary(String filen)
    {
        fileHandle = new File(filen);
        try {
            writer = new FileWriter(fileHandle);
        } catch (Exception e)
        {
            
        }
    }
    protected void finalize() throws Throwable {
        try {

        } finally {
            super.finalize();
        }
    }

    public void writeDictionary(String word, String define)
    {
        try {
            writer.append(word + " : " + define);
        } catch (Exception e) {
            System.out.println("Write error");
        }
    }

    public void readDictionary()
    {
        boolean validFile;
        do {
            Scanner inputFile = null;
            validFile = true;
            try {
            inputFile = new Scanner (fileHandle);
            } catch (Exception e) {
            System.out.println("Error opening input file: ");
            validFile = false;
            }
        } while (validFile);

    }

    public void close()
    {
        try {
           writer.close();
        } catch (Exception e) {

        }
    }
}