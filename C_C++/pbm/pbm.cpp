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
#include <cstdio>

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

static const char binary[16][5] =
{ "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000",
"1001", "1010", "1011", "1100", "1101", "1110", "1111" };

static const uint8_t BitReverseTable256[] =
{
  0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
  0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
  0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
  0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
  0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
  0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
  0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
  0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
  0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
  0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
  0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
  0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
  0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
  0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
  0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
  0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

uint8_t * Pbm::data()
{
	uint8_t * tmpData = new uint8_t[m_pixelsArrSize];
	memcpy(tmpData, m_pixels, m_pixelsArrSize);
	return tmpData;
}

uint8_t * Pbm::dataHflip()
{
    uint8_t * tmpData = new uint8_t[m_pixelsArrSize];
    int y = 0, arrWidth = m_width/8;
    for (int i = 0; i < m_pixelsArrSize; i++)
    {
        // Flip bytes in a row
        tmpData[i] = m_pixels[(arrWidth-i%arrWidth - 1) + y*arrWidth];
        (i+1) % arrWidth == 0 ? y++ : y;
    }
    for (int i = 0; i < m_pixelsArrSize; i++)
    {
        // Flip bits in a byte
        tmpData[i] = BitReverseTable256[tmpData[i]];
    }
    return tmpData;
}


void Pbm::hex2ascii(char * ascii_arr, char * byte_arr, int length_byte_arr)
{
	std::cout << "len: " << length_byte_arr << std::endl;
	char nextAsciiByte[9] = "";
	for (int i = 0; i < length_byte_arr; i++)
	{
		nextAsciiByte[0] = '\0';
		strcat(nextAsciiByte, binary[(byte_arr[i] >> 4) & 0xF]); // four bits
		strcat(nextAsciiByte, binary[byte_arr[i] & 0xF]); // four bits

		memcpy(ascii_arr, nextAsciiByte, 8);
		ascii_arr += 8; // move forward 8 bit characters
	}
}

void Pbm::ascii2hex(uint8_t * byte_arr, char * ascii_arr)
{
	/* Internally used. Just ascii 1s & 0s. to hex. no outside characters allowed!
	Expects large amounts of data in multiples of 8. Expected null terminated

	Ex. len("01011010") =8  ---->  0x5A (1 byte)
	*/

	int remainder_bits = strlen(ascii_arr) % (sizeof(unsigned long) * 8);
	int num_long_bytes = strlen(ascii_arr) / (sizeof(unsigned long) * 8);
	//char * endPtr = ascii_arr;

	unsigned long bits;
	char substring[sizeof(unsigned long) * 8 + 1];

	for (int i = 0; i < num_long_bytes; i++)
	{
		memcpy(substring, ascii_arr, sizeof(unsigned long) * 8);
		substring[sizeof(unsigned long) * 8] = '\0';

		//std::cout << "subs: " << substring << std::endl;
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
	if (remainder_bits > 0)
	{
		memcpy(substring, ascii_arr, remainder_bits);
		substring[remainder_bits] = '\0';
		std::bitset<32> xbits(substring);
		//std::cout << "xbits: " << xbits << std::endl;
		//std::cout << "remainder_bits: " << remainder_bits << std::endl;
		//std::cout << "shifting: " << (sizeof(unsigned long)*8 - remainder_bits) << std::endl;
		xbits <<= (sizeof(unsigned long) * 8 - remainder_bits);
		//std::cout << "xbits: " << xbits << std::endl;
		bits = xbits.to_ulong();
		//std::cout << "xbits long: " << bits << std::endl;
		for (int i = 0; i < remainder_bits / 8; i++)
		{
			//std::cout << (bits >> (sizeof(unsigned long)  - i -1)*8) << std::endl;
			byte_arr[i] = bits >> ((sizeof(unsigned long) - i - 1) * 8) & 0xff;
			//printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(byte_arr[i]));

		}
		printf("\n");
	}
}

Pbm::Pbm(int width, int height) :
	m_height(height), m_width(width)

{
	m_imageSize = m_width * height;

	// 8 Pixels stored in each byte.
	m_pixelsArrSize = m_imageSize / 8;
	m_pixels = new uint8_t[m_pixelsArrSize];
	memset(m_pixels, 0, m_pixelsArrSize);

}
void Pbm::print()
{
	// Testing
	std::cout << "\n\tImage\n" << std::endl;
	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width / 8; j++)
		{
		    if (m_pixels[i * (m_width / 8) + j] < 0x40)
		        std::cout << '.';
		    else if (m_pixels[i * (m_width / 8) + j] < 0x80)
		        std::cout << ':';
		    else if (m_pixels[i * (m_width / 8) + j] < 0xC0)
		        std::cout << '?';
		    else
		        std::cout << '0';
		}
		std::cout << std::endl;
	}
}

Pbm::Pbm(const char* filename)
{
	std::ifstream file;
	file.open(filename, std::ios::binary);		//|std::ios_base::skipws);
	uint16_t magic_number = UNK_MAGIC_NUMBER;
	char * ascii_row;
	int real_width;
	//std::string ascii_data;
	m_pixels = nullptr;
	uint8_t * ppixels = m_pixels;

	if (file.is_open())
	{
		// load file header
		file.read((char *)&magic_number, sizeof(magic_number));
		printf("0x%x\n", magic_number);
		if (file.peek() == '\n')
			file.get();
		if (file.peek() == '#')
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		// Make sure is a ascii pbm file
		if (magic_number == PLAIN_MAGIC_NUMBER)
		{
			file >> real_width; //m_width;
			file >> m_height;
			printf("width: %d\n", real_width); //m_width
			printf("height: %d\n", m_height);

			m_width = (real_width % 8 == 0) ? real_width : real_width + 8 - real_width % 8;
			if (m_width % 8 == 0)
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
			m_pixelsArrSize = m_imageSize / 8;
			m_pixels = new uint8_t[m_pixelsArrSize];
			ascii_row = new char[m_width + 1](); // value initialize!

			std::string ascii_data((std::istreambuf_iterator<char>(file)),
				std::istreambuf_iterator<char>());

			ascii_data.erase(
				std::remove(ascii_data.begin(), ascii_data.end(), '\r'),
				ascii_data.end());
			ascii_data.erase(
				std::remove(ascii_data.begin(), ascii_data.end(), '\n'),
				ascii_data.end());

			std::cout << "data length: " << ascii_data.length() << std::endl;
			ppixels = m_pixels;

			printf("pixel array size: %d\n", m_pixelsArrSize);
			printf("real_width, m_height:  %d, %d\n", real_width, m_height);

			for (int i = 0; i < m_height; i++)
			{
				ascii_data.copy(ascii_row, real_width, i * real_width); //(arr, size, start_pos)
				strcat(ascii_row, std::string("00000000").substr(0, (m_width - real_width)).c_str());
				ascii_row[m_width] = '\0';
				ascii2hex(ppixels, ascii_row);
				ppixels += m_width / 8;
				// Set to m_pixels
			}

		} // end ascii
		else if (magic_number == MAGIC_NUMBER)
		{
			file >> real_width;
			file >> m_height;
			printf("width: %d\n", real_width);
			printf("height: %d\n", m_height);
			m_width = (real_width % 8 == 0) ? real_width : real_width + 8 - real_width % 8;
			if (file.peek() == '\n')
				file.get(); // printf("got the turd");

			/*if (m_width == 384)
			{
				printf("Width is correct\n");
				m_imageSize = m_width * m_height;
			}
			else*/ if (m_width % 8 == 0)
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
			m_pixelsArrSize = m_imageSize / 8;
			m_pixels = new uint8_t[m_pixelsArrSize] ();
			std::cout << m_width << " " << m_height << std::endl;
			// start replace
			int real_bytes_len = real_width / 8;
			printf("gettin data %d  %d\n", m_width / 8, real_bytes_len);
			ppixels = m_pixels;

			while (!file.eof())
			{
				file.read((char*)ppixels, real_bytes_len);
				ppixels += real_bytes_len;
			}
			// replaces
			//file.read((char*)m_pixels, m_pixelsArrSize);
			// end
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
	/*
	Saving in P4 pbm format is default
	*/
	std::ofstream file;
	file.open(filename, std::ios::binary);

	if (file.is_open())
	{
		file << "P4" << std::endl;
		file << m_width << " " << m_height << std::endl;
		file.write((char *)m_pixels, m_pixelsArrSize);
		file.close();
	}
	else
	{
		return false;
	}
	return true;
}

bool Pbm::save(const char *filename, const uint8_t * data, const int data_len,
	const int data_width)
{
	/*
	Saving in P4 pbm format is default
	*/

	if (data_len % (data_width / 8) != 0)
	{
		std::cerr << "Improper length or width" << std::endl;
		return false;
	}
	std::ofstream file;
	file.open(filename, std::ios::binary);

	if (file.is_open())
	{
		file << "P4" << std::endl;
		file << data_width << " " << data_len * 8 / data_width << std::endl;
		file.write((char *)data, data_len);
		file.close();
	}
	else
	{
		return false;
	}
	return true;
}

bool Pbm::save(const char *filename, const char * type)
{
	uint16_t magic_number = UNK_MAGIC_NUMBER;

	memcpy(&magic_number, type, 2);

	// Make sure is a ascii pbm file
	if (magic_number == PLAIN_MAGIC_NUMBER
		|| std::string("ascii").compare(type) == 0
		|| std::string("plain").compare(type) == 0)
	{
		std::cout << "Plain" << std::endl;

		std::ofstream file;
		file.open(filename);

		if (file.is_open())
		{
			char * out_data = new char[m_width * m_height + 1];

			file << "P1" << std::endl;
			file << m_width << " " << m_height << std::endl;
			hex2ascii(out_data, (char*)m_pixels, m_pixelsArrSize);
			out_data[m_width * m_height] = '\0';
			file.write(out_data, m_width * m_height);

			file.close();
			delete[] out_data;

		}
		else
		{
			return false;
		}
	}
	else if (magic_number == MAGIC_NUMBER
		|| std::string("binary").compare(type) == 0
		|| std::string("hex").compare(type) == 0)
	{
		std::cout << "binary" << std::endl;
		return save(filename);

	}
	else
	{
		std::cerr << "Not a valid pbm filetype" << std::endl;
	}
	return true;
}

bool Pbm::pad(const int new_width)
{
	printf("Begin padding...");

	int new_width8;
	new_width8 = (new_width % 8 == 0) ? new_width : new_width + 8 - new_width % 8;

	int old_width = m_width;
	int width_growth = new_width8 - old_width;
	uint8_t * tmp_buffer, *tmpPtr;
	uint8_t * pixelPtr;
	assert(width_growth % 8 == 0);
	//assert(width_growth > 0);
	if (width_growth > 0 && m_pixels != nullptr)
	{
		tmp_buffer = new uint8_t[m_pixelsArrSize] ();
		memcpy(tmp_buffer, m_pixels, m_pixelsArrSize);
		tmpPtr = tmp_buffer;
		delete[] m_pixels;

		m_width = new_width8;
		m_imageSize = m_width * m_height;
		m_pixelsArrSize = m_imageSize / 8;
		m_pixels = new uint8_t[m_pixelsArrSize];
		pixelPtr = m_pixels;
		printf("New width %d", m_width);

		int bytes_growth = width_growth / 8;
		int pad_size;
		for (int y = 0; y < m_height; y++)
		{
			// Padding
			pad_size = bytes_growth / 2 + bytes_growth % 2;
			memset(pixelPtr, 0, pad_size);
			pixelPtr += pad_size;

			// Copy over original data
			memcpy(pixelPtr, tmpPtr, old_width / 8);
			pixelPtr += old_width / 8;
			tmpPtr += old_width/8;

			// Padding
			pad_size = bytes_growth / 2;
			memset(pixelPtr, 0, pad_size);
			pixelPtr += pad_size;
		}

		delete[] tmp_buffer;
		printf("Grew image!");

	}
	else if (width_growth == 0)
	{
		printf("No growth");
		return true;
	}
	else if (m_pixels == nullptr)
	{
		std::cerr << "Image not initialized.";
	}
	else if (width_growth < 0)
	{
		std::cerr << "Shrinking image not yet supported. New W must be > than old width.";
	}
	else
	{
		std::cerr << "Pbm::pad Unknown Error";
		return false;
	}
	return true;
}

value Pbm::pixel(int x, int y)
{
	int ret_value = -1;
	if (x >= 0 && y >= 0 && x < m_width && y < m_height)
	{
		int byte_index = (y * m_width + x) / 8;
		int bit_index = x % 8;
		ret_value = (m_pixels[byte_index] >> (7 - bit_index)) & 0x01;
	}
	else
		std::cerr << "Pixel out of bounds\n" << std::endl;
	return ret_value;
}

void Pbm::setPixel(int x, int y, value bw)
{
	if (x >= 0 && y >= 0 && x < m_width && y < m_height)
	{
		int byte_index = (y * m_width + x) / 8;
		int bit_index = x % 8;
		m_pixels[byte_index] |= (bw << (7 - bit_index));
	}
	else
		std::cerr << "Pixel out of bounds\n" << std::endl;
}

int Pbm::row(int y, uint8_t* buffer_bits, int buffer_len)
{
	int start_index = y * m_width / 8;

	if (y >= m_height)
	{
		std::cerr << "Requested pixel row out of bounds" << std::endl;
		return 0;
	}
	// Don't copy beyond end of row
	if (buffer_len > m_width / 8)
		buffer_len = m_width / 8;

	memcpy(buffer_bits, &m_pixels[start_index], buffer_len);
	return buffer_len * 8;
}

int Pbm::setRow(int y, uint8_t* data_bits, int data_len)
{
	int start_index = y * m_width / 8;

	if (y >= m_height)
	{
		std::cerr << "Requested pixel row out of bounds" << std::endl;
		return 0;
	}

	// Don't copy beyond end of row
	if (data_len > m_width / 8)
		data_len = m_width / 8;

	memcpy(&m_pixels[start_index], data_bits, data_len);

	if (data_len < m_width / 8)
	{
		std::cerr << "Not a complete row.." << std::endl;
		return data_len * 8;
	}
	else if (data_len == m_width / 8)
	{
		return m_width;
	}
	return 0;
}
