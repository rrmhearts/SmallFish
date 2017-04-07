/* 
 * File:   main.cpp
 * Author: Goshikku
 *
 * Created on May 19, 2010, 8:13 PM
 */

#include <stdlib.h>
#include <stdio.h>

/*
 * 
 */
int * stack2()
{
    int d = 1;
    int * pd = &d;
    printf("stack location: %x\n", &d);
    return pd;
}

int main()
{
    int here = 100;
    int* address = stack2();
    int* temp = &here;
    printf("my location: %x\n", &here);
    int i = 0;
    for (i =0; i < 20; i++)
    {
        if (*temp == *address)
        {
            printf("Here! %d, %x\n", i, temp);
            break;
        }
        temp++;        
    }
    printf("Success, %x should be %x\n", (&here-0x10), address);
    int k = 0x22fee4;
    int * j = (int *)k;
    printf("ans: %d, %d", *j, *address);

    return 0;
}
/*
char * stack1()
{
    char* s2loc = stack2();
    char c = '1';
    char * pc = &c;
    pc = (char*)(s2loc);
    printf("pc-11: %d\n%d\n%x\n", *pc, *s2loc, &c-s2loc);
    //*(pc) = 7;
    return (char*)&c+1;
}
int main(int argc, char** argv) {

    char * st1,* st2;
    printf("m:stack1addr: %x\n", st1=stack1());
    char d;
    char *pd = &d;
    printf("m:mainst1diff: %x\n", pd-st1);
    printf("mainloc: %x\n", pd);
    return (EXIT_SUCCESS);
}*/

