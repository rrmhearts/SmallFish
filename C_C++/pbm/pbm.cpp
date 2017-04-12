/*
 * pbm.cpp
 *
 *  Created on: Apr 12, 2017
 *      Author: Ryan McCoppin
 */

#include "pbm.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <bitset>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

void ascii2hex(uint8_t * byte_arr, char * ascii_arr)
{

	int remainder_bytes = strlen(ascii_arr) % (sizeof(unsigned long)*8);
	int num_long_bytes = strlen(ascii_arr) / (sizeof(unsigned long) *8);
	//char * endPtr = ascii_arr;
	unsigned long bits;
	char substring[sizeof(unsigned long)*8+1];

	for (int i = 0; i < num_long_bytes; i++)
	{		
		strncpy_s(substring, ascii_arr, sizeof(unsigned long) * 8);
		substring[sizeof(unsigned long)*8] = '\0';

		std::bitset<32> xbits(substring);
		bits = xbits.to_ulong();

		// Assign long bits to byte array
		byte_arr[0] = (bits >> 24) & 0xff; 
		byte_arr[1] = (bits >> 16) & 0xff;
		byte_arr[2] = (bits >> 8) & 0xff; 
		byte_arr[3] = bits & 0xff;

		byte_arr += sizeof(unsigned long);
		ascii_arr += sizeof(unsigned long) * 8;
	}

	strncpy_s(substring, ascii_arr, remainder_bytes);
	substring[remainder_bytes] = '\0';
	std::bitset<32> xbits(substring);
	bits = xbits.to_ulong();
	for (int i = 0; i < remainder_bytes; i++)
	{
		byte_arr[i] = bits >> (sizeof(unsigned long) - i - 1) * 8 & 0xff;
	}
}
Pbm::Pbm(int width, int height)
:  m_height(height), m_width(width)

{
    m_imageSize = m_width * height;

    // 8 Pixels stored in each byte.
    m_pixelsArrSize = m_imageSize / 8;
    m_pixels = new uint8_t[m_pixelsArrSize];
    memset(m_pixels, 0, m_pixelsArrSize);

}

Pbm::Pbm(int width, int height, const char* type)
{
}

Pbm::Pbm(const char* filename)
{
    std::ifstream file;
    file.open(filename, std::ios::binary|std::ios_base::skipws);
    uint16_t magic_number = UNK_MAGIC_NUMBER;
	char * ascii_row;
	//std::string ascii_data;
	m_pixels = NULL;
    if (file.is_open())
    {
        // load file header
        file.read((char *)&magic_number, sizeof(magic_number));
		printf("0x%x\n", magic_number);
        // Make sure is a ascii pbm file
        if (magic_number == PLAIN_MAGIC_NUMBER)
        {
            file >> m_width;
            file >> m_height;
			printf("width: %d\n", m_width);
			printf("height: %d\n", m_height);

            if (m_width == 384)
            {
                printf("Width is correct\n");
                m_imageSize = m_width * m_height;
            }
			else if (m_width % 8 == 0)
			{
				printf("Width is acceptable, divisible by 8: %d\n", m_width);
				m_imageSize = m_width * m_height;
			}
			else
            {
                std::cerr << "Invalid pbm header " << std::endl;
                m_height = 0;
                m_width = 0;
                m_imageSize = 0;
            }

            // Read bitmap data
            m_pixelsArrSize = m_imageSize/8;
            m_pixels = new uint8_t[m_pixelsArrSize];
			ascii_row = new char[m_width + 1];
			
			std::string ascii_data((std::istreambuf_iterator<char>(file)), 
				std::istreambuf_iterator<char>());

			ascii_data.erase(std::remove(ascii_data.begin(), ascii_data.end(), '\n'), ascii_data.end());

			std::cout << "data length: " << ascii_data.length() << std::endl;
			uint8_t * ppixels = m_pixels;
			for (int i = 0; i < m_height; i++)
			{
				ascii_data.copy(ascii_row, m_width, i*m_width);
				ascii_row[m_width] = '\0';
				//std::cout << ascii_row << std::endl;
				ascii2hex(ppixels, ascii_row);
				ppixels += m_width/8;
				// Set to m_pixels
			}			
			for (int i = 0; i < m_height; i++)
			{
				for (int j = 0; j < m_width / 8; j++)
				{
					std::cout << (m_pixels[i*(m_width/8)+j] > 0);
				}
				std::cout << std::endl;
			}
			for (int i = 122880; i < 122880; i++)
			{
				printf("pix @ (%d,%d): %d ",i%384,i/384, pixel(i % 384, i/384));

				if (i % 8 == 0)
				{
					printf("pixels %d: %d\n", i/8, m_pixels[i/8]);
				}
				else
					printf("\n");
			}

        }
        else if (magic_number == MAGIC_NUMBER){

        }
        else
        {
            std::cerr << "Not a valid pbm file" << std::endl;
            m_height = 0;
            m_width = 0;
            m_imageSize = 0;
        }

        file.close();
    }
    else
    {
        std::cerr << "Failed to open " << std::endl;
        m_height = 0;
        m_width = 0;
        m_imageSize = 0;
    }
}

bool Pbm::save(const char* filename)
{
	return true;
}

value Pbm::pixel(int x, int y)
{
	int ret_value = -1;
	if (x >= 0 && y >= 0 && x < m_width && y < m_height)
	{
		int byte_index = (y*m_width + x) / 8;
		int bit_index = x % 8;
		ret_value = (m_pixels[byte_index] >> (7-bit_index)) & 0x01;
	}
	else
		std::cerr << "Pixel out of bounds\n" << std::endl;
	return ret_value;
}

void Pbm::setPixel(int x, int y, value bw)
{

}

int Pbm::row(int y, uint8_t* buffer, int len)
{
	return 0;
}



int Pbm::setRow(int y, uint8_t* data, int len)
{
	return 0;
}
