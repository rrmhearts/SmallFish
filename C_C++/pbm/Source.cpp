#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "pbm.h"

/*
Ryan McCoppin

*/
const char binary[16][5] = { "0000", "0001", "0010", "0011",
"0100", "0101", "0110", "0111",
"1000", "1001", "1010", "1011",
"1100", "1101", "1110", "1111"
};
const int dimx = 24, dimy = 24;

// Out file pbm
FILE *pbm;//, *expand;
FILE * expand_hex;



float scale_g = 1.0; // only in old main
char * inputhex = "bitmap.hex";

char * hex2pbm(char * line)
{

	int i = 0, ref = 0;
	for (i = 0; i < 6; i++)
	{
		if (line[i] <= '9' && line[i] >= '0')
		{
			ref = line[i] - '0';
			fprintf(pbm, "%s", binary[ref]);
		}
		else if (line[i] <= 'F' && line[i] >= 'A')
		{
			ref = line[i] - 'A' + 10;
			fprintf(pbm, "%s", binary[ref]);
		}

	}
	fprintf(pbm, "\n");
	return 0;
}

void pbm2hex()
{
	char substring[5];
	char hex[128];
	char line[128];
	char line_buffer[256];
	//printf("line  %s\n", line);
	int i = 0;
	int len = 90;
	hex[127] = '\0';
	
	FILE * input_pbm;
	//fopen_s(&input_pbm, "ache_letter_pbm.pbm", "r"); // Read expanded data to trasit to hex
	fopen_s(&input_pbm, "narrow_gap_Hs_less_spacing.pbm", "r"); // Read expanded data to trasit to hex
	FILE * output_hex;
	fopen_s(&output_hex, "littleache.hex", "w"); // hex version of binary expanded data
	printf("Opened files...");
	fgets(line, len, input_pbm);
	fgets(line, len, input_pbm); 
	printf("len: %u", (unsigned)strlen(line));
	char * pline = NULL;
	while (fgets(line, len, input_pbm) != NULL) {
		char substring[5];
		char hex[128];
		char leftovers[20];
		i = 0; hex[127] = '\0';
		size_t ln = strlen(line) - 1;
		if (line[ln] == '\n')
			line[ln] = '\0'; // remove new line

		// What to do with line leftovers?
		memset(leftovers, 0, 20);
		if (pline != NULL && (pline[0] == '1' || pline[0] == '0') )
		{
			printf("leftov pline: %s\n", pline);
			strcpy_s(leftovers, pline);
			printf("leftov: %s\n", pline);
			
		}
		memset(line_buffer, 0, 256);
		strcat_s(line_buffer, leftovers); // leftover of previous line
		strcat_s(line_buffer, line); // next line
		pline = (char *) line_buffer; // was line
		printf("line: %s\n", line);
		printf("lbuff: %s\n", line_buffer);
		printf("pline: %s\n", pline);
		//pline = (char *)line;
		while (pline[7] == '1' || pline[7] == '0')
		{
			strncpy_s(substring, pline, 4);
			substring[4] = '\0';
			//printf("sub %s\n", substring);
			//printf("%s\n",substring);
			if (i % 2 == 0)
			{
				hex[i] = '0';
				i++;
				hex[i] = 'x';
				i++;
			}
			if (!strcmp(substring, "0000")) {
				hex[i] = '0';
			}
			else if (!strcmp(substring, "0001")) {
				hex[i] = '1';
			}
			else if (!strcmp(substring, "0010")) {
				hex[i] = '2';
			}
			else if (!strcmp(substring, "0011")) {
				hex[i] = '3';
			}
			else if (!strcmp(substring, "0100")) {
				hex[i] = '4';
			}
			else if (!strcmp(substring, "0101")) {
				hex[i] = '5';
			}
			else if (!strcmp(substring, "0110")) {
				hex[i] = '6';
			}
			else if (!strcmp(substring, "0111")) {
				hex[i] = '7';
			}
			else if (!strcmp(substring, "1000")) {
				hex[i] = '8';
			}
			else if (!strcmp(substring, "1001")) {
				hex[i] = '9';
			}
			else if (!strcmp(substring, "1010")) {
				hex[i] = 'A';
			}
			else if (!strcmp(substring, "1011")) {
				hex[i] = 'B';
			}
			else if (!strcmp(substring, "1100")) {
				hex[i] = 'C';
			}
			else if (!strcmp(substring, "1101")) {
				hex[i] = 'D';
			}
			else if (!strcmp(substring, "1110")) {
				hex[i] = 'E';
			}
			else if (!strcmp(substring, "1111")) {
				hex[i] = 'F';
			}
			i++;
			pline += 4;

			strncpy_s(substring, pline, 4);
			substring[4] = '\0';

			if (!strcmp(substring, "0000")) {
				hex[i] = '0';
			}
			else if (!strcmp(substring, "0001")) {
				hex[i] = '1';
			}
			else if (!strcmp(substring, "0010")) {
				hex[i] = '2';
			}
			else if (!strcmp(substring, "0011")) {
				hex[i] = '3';
			}
			else if (!strcmp(substring, "0100")) {
				hex[i] = '4';
			}
			else if (!strcmp(substring, "0101")) {
				hex[i] = '5';
			}
			else if (!strcmp(substring, "0110")) {
				hex[i] = '6';
			}
			else if (!strcmp(substring, "0111")) {
				hex[i] = '7';
			}
			else if (!strcmp(substring, "1000")) {
				hex[i] = '8';
			}
			else if (!strcmp(substring, "1001")) {
				hex[i] = '9';
			}
			else if (!strcmp(substring, "1010")) {
				hex[i] = 'A';
			}
			else if (!strcmp(substring, "1011")) {
				hex[i] = 'B';
			}
			else if (!strcmp(substring, "1100")) {
				hex[i] = 'C';
			}
			else if (!strcmp(substring, "1101")) {
				hex[i] = 'D';
			}
			else if (!strcmp(substring, "1110")) {
				hex[i] = 'E';
			}
			else if (!strcmp(substring, "1111")) {
				hex[i] = 'F';
			}
			i++;

			if (i % 2 == 0)
			{
				hex[i] = ',';
				i++;
				hex[i] = ' ';
				i++;
			}
			pline += 4;
		} // end while..
		

		hex[i] = '\0';
		fprintf(output_hex, "%s", hex);
		fprintf(output_hex, "\n");
	}
	fclose(output_hex);
	
	//printf("%s,    %d\n", hex, i);
	//return hex;
}


int main(int argc, char* argv[])
{

	/*float scale_x = 2, scale_y = 2;
	char * hex = "bitmap.hex";
	if (argc >= 4) {
		char * hex = argv[1];
		scale_x = atoi(argv[2]);
		scale_y = atoi(argv[3]);
	}*/
	//pbm2hex();

	//Pbm obj = Pbm("bob.pbm");
	//Pbm obj = Pbm("ache_letter_pbm.pbm");
	Pbm obj = Pbm("ache_letter_pbm_300x250_raw.pbm");
	obj.print();
	getchar();

	obj.pad(384);
	obj.print();
	// Pbm("remainder_pbm_b.pbm");
	//obj.setPixel(0, 0, 0);
	//char * buffer = new char[1000];
	/*uint8_t buffer[1000];
	for (int i = 0; i < obj.height(); i++)
	{
		obj.row(i, buffer, 1000);

		for (int x = 0; x < obj.width()/8; x++)
		{
			printf("%x", buffer[x]);
		}
		printf("\n");

	}*/
	//obj.setPixel(1, 1, 0);
	//obj.save("bob.pbm", "ascii");
	//obj.save("bob2.pbm", );
	//setupGraphicFont(scale_x, scale_y);
	getchar();
	getchar();
	getchar();

	return 0;
}
