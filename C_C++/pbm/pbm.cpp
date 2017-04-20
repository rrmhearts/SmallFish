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

const char binary[16][5] =
{ "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000",
"1001", "1010", "1011", "1100", "1101", "1110", "1111" };

uint8_t * Pbm::data()
{
	uint8_t * tmpData = new uint8_t[m_pixelsArrSize];
	memcpy(tmpData, m_pixels, m_pixelsArrSize);
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
			m_pixels = new uint8_t[m_pixelsArrSize]();
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
		tmp_buffer = new uint8_t[m_pixelsArrSize]();
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
			tmpPtr += old_width / 8;

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
