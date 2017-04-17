/*
 * pbm.h
 *
 *  Created on: Apr 12, 2017
 *      Author: Ryan McCoppin
 */

#ifndef SRC_PBM_H_
#define SRC_PBM_H_

#include <cstdint>

typedef uint8_t value;

static const uint16_t MAGIC_NUMBER = 0x3450; // "P4"
static const uint16_t PLAIN_MAGIC_NUMBER = 0x3150; // "P1"
static const uint16_t UNK_MAGIC_NUMBER = 0x3050; // P null

class Pbm
{
    public:
        Pbm(int width, int height);
        Pbm(const char *filename);
        virtual ~Pbm()
        {
            if (m_pixels != nullptr)
                delete[] m_pixels;
        }

        bool save(const char *filename);
        bool save(const char *filename, const uint8_t * data,
                const int data_len, const int data_width); // save external data
        bool save(const char *filename, const char * type);

        uint8_t * data();
        bool pad(const int new_width);

        // In pixels
        int height()
        {
            return m_height;
        }
        int width()
        {
            return m_width;
        }

        value pixel(int x, int y);
        void setPixel(int x, int y, value bw);

        // get a whole row of bytes, 8 pixels per byte
        int row(int y, uint8_t *buffer, int len);
        // set a whole row of bytes
        int setRow(int y, uint8_t *data, int len);

        // internally used.
        void ascii2hex(uint8_t * byte_arr, char * ascii_arr);
        void hex2ascii(char * ascii_arr, char * byte_arr, int length_byte_arr);

    private:
        int m_height;
        int m_width;// These vars are in pixel space. Height and width of image
        int m_imageSize;   // total # of pixels

        int m_pixelsArrSize; // These vars are in representation space. One bit per pixel. Effectively 1/8 the size of pixel space vars.
        uint8_t *m_pixels;   // 8 pixels per byte

};

#endif /* SRC_PBM_H_ */
