#include <stdint.h>

/* Función requerida para reemplazar la implementada en assembler en la biblioteca CMSIS-DSP 
	
	Principalmente mueve los elementos de un array de forma tal que los índices tengan "flipeados" sus bits,
	debido a cómo calcula la FFT los valores.
	
	Extraído de https://community.arm.com/thread/9182
*/

void arm_bitreversal_32 (uint32_t * pSrc, const uint16_t bitRevLen, const uint16_t * pBitRevTable){  
  uint32_t r7,r6,r5,r4,r3;  
  if (bitRevLen <= 0)  
    return;  
  r3 = ((bitRevLen+1) >> 2);  
  while (r3 > 0)  
    {  
      r7 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[3]);  
      r6 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[2]);  
      r5 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[1]);  
      r4 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[0]);  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[3]) = r6;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[2]) = r7;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[1]) = r4;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[0]) = r5;  
      r7 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[3] + 4);  
      r6 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[2] + 4);  
      r5 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[1] + 4);  
      r4 = *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[0] + 4);  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[3] + 4) = r6;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[2] + 4) = r7;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[1] + 4) = r4;  
      *(uint32_t*)((uint8_t*) pSrc + pBitRevTable[0] + 4) = r5;  
      pBitRevTable += 4;  
      r3--;  
    }  
}  



// void arm_bitreversal_32(uint32_t *pSrc, const uint16_t bitRevLen,
//       const uint16_t *pBitRevTab) {
//         uint32_t a, b, i, tmp;

//         for (i = 0; i < bitRevLen;) {
//             a = pBitRevTab[i] >> 2;
//             b = pBitRevTab[i + 1] >> 2;

//             // real
//             tmp = pSrc[a];
//             pSrc[a] = pSrc[b];
//             pSrc[b] = tmp;

//             // complex
//             tmp = pSrc[a + 1];
//             pSrc[a + 1] = pSrc[b + 1];
//             pSrc[b + 1] = tmp;

//             i += 2;
//         }
// }