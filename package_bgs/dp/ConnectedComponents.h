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
/****************************************************************************
*
* ConnectedComponents.h
*
* Purpose: 	Find connected components in an image. This class effectively just
*			encapsulates functionality found in cvBlobLib.
*
* Author: Donovan Parks, August 2007
*
******************************************************************************/

#ifndef _CONNECTED_COMPONENTS_H_
#define _CONNECTED_COMPONENTS_H_

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "BlobResult.h"
#include "Image.h"

class ConnectedComponents
{
public:
  ConnectedComponents(){}
  ~ConnectedComponents(){}
  
  void SetImage(BwImage* image) { m_image = image; }

  void Find(int threshold);

  void FilterMinArea(int area, CBlobResult& largeBlobs);

  void GetBlobImage(RgbImage& blobImage);
  void SaveBlobImage(char* filename);

  void GetComponents(RgbImage& blobImage);

  void FilterSaliency(BwImage& highThreshold, RgbImage& blobImg, float minSaliency, 
                          CBlobResult& salientBlobs, CBlobResult& unsalientBlobs);

  void ColorBlobs(IplImage* image, CBlobResult& blobs, CvScalar& color);

private:
  BwImage* m_image;

  CBlobResult m_blobs;
};

#endif
