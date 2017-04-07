package javaapplication1;
import java.util.Random;

/**
 *
 * @author Goshikku
 
class RSA {

    int p, q, n, phi, e, d;
    public RSA ()
    {
        p = 5;
        q = 3;
        n = p*q;
        phi = (p-1)*(q-1);
        e = 7;
        d = 12;

    }
    public int encrypt(int m)
    {
        return m^e % n;
    }
    public int decrypt(int c)
    {
        return c^d % n;
    }
}
 *
 */
public class Main {
    public static int inverse(int a, int n)
    {
        int temp, t = 1, to = 0, b, q, r;
        b = n;
        q = b/a;
        r = b - q*a;
        while (r > 0)
        {
            temp = to - q*t;
            temp = temp % n;
            b = a;
            a = r;
            q = b/a;
            r = b - q*a;
            to = t;
            t = temp;
        }
        if ( a != 1)
        {
            return 0;
        } else {
            return t;
        }
    }
/*
 * s = f(0) = Sum yi Mult (-j)(i-j)^-1
 *             i     j!=i
 */
    public static int getS(int[] x, int[] y, int n, int the_x)
    {
        int s = 0, mult;
        for (int i = 1; i < x.length; i++)
        {
            mult = 1;
            for (int j = 1; j < x.length; j++)
            {
                if (j != i && x[i]-x[j] !=0)
                {
                    mult = mult * the_x-x[j] * inverse(x[i]-x[j], n) % n;
                }
            }
            //System.out.println("S: " + s + ", mult: " + mult);
            mult = mult * y[i] % n;
            s = (s + mult) % n;
        }
        //System.out.println("S: " + s);
        return s;
    }
    public static void main(String[] args) {

        int[] x = {413, 432, 451, 470, 489, 508, 527, 546, 565, 584};
        int[] y = {25439, 14847, 24780, 5910, 12734, 12492, 12555, 28578,
            20806, 21462};

        int n = 10, s;
        int p = 31847;
        
        Random r = new Random();
        int rand;
        int[] sx;
        int[] sy;
      /*  int randomL;
        do {
            randomL= r.nextInt() % x.length;
            if (randomL < 0)
                randomL = -randomL;
        } while (randomL < x.length/2);*/

        sx = new int[5];
        sy = new int[5];

        rand = (r.nextInt() % x.length + x.length) % x.length;
        for (int i = 0; i < 5; i++)
        {
            sx[i] = x[rand];
            sy[i] = y[rand];
            rand = ++rand % x.length;// (r.nextInt() % x.length + x.length) % x.length;
        }
        s = getS(sx, sy, p, 10000);//0
        System.out.println(s);
    }

}