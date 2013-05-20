/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
//***********************************************************//
//* Blob analysis package  8 August 2003                    *//
//* Version 1.0                                             *//
//* Input: IplImage* binary image                           *//
//* Output: attributes of each connected region             *//
//* Author: Dave Grossman                                   *//
//* Email: dgrossman@cdr.stanford.edu                       *//
//* Acknowledgement: the algorithm has been around > 20 yrs *//
//***********************************************************//


#if !defined(_CLASSE_BLOBEXTRACTION_INCLUDED)
#define _CLASSE_BLOBEXTRACTION_INCLUDED


//! Extreu els blobs d'una imatge
bool BlobAnalysis(IplImage* inputImage, uchar threshold, IplImage* maskImage,
                  bool borderColor, bool findmoments, blob_vector &RegionData );


// FUNCIONS AUXILIARS

//! Fusiona dos blobs
void Subsume(blob_vector &RegionData, int, int*, CBlob*, CBlob*, bool, int, int );
//! Reallocata el vector auxiliar de blobs subsumats
int *NewSubsume(int *SubSumedRegion, int elems_inbuffer);
//! Retorna el perimetre extern d'una run lenght
double GetExternPerimeter( int start, int end, int row, int width, int height, IplImage *maskImage );

#endif //_CLASSE_BLOBEXTRACTION_INCLUDED
