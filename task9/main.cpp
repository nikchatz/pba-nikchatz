
#include "delfem2/glfw/viewer2.h"
#include "delfem2/mshprimitive.h"
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/SVD>
//#include <Eigen/LU>

/**
 * making 2D regular grid with edge length 1
 * @tparam REAL float or double
 * @param aXYZ list of points (XYXYXY...)
 * @param aQuad vertex index for quad mesh
 * @param nx nubmer of quads in x direction
 * @param ny number of quads in y direction
 */
template <typename REAL>
void MeshQuad2_Grid(
    std::vector<REAL>& aXYZ,
    std::vector<unsigned int>& aQuad,
    unsigned int nx,
    unsigned int ny)
{
  unsigned int np = (nx+1)*(ny+1);
  aXYZ.resize(np*2);
  for(unsigned int iy=0;iy<ny+1;++iy){
    for(unsigned int ix=0;ix<nx+1;++ix){
      unsigned int ip = iy*(nx+1)+ix;
      aXYZ[ip*2+0] = ix;
      aXYZ[ip*2+1] = iy;
    }
  }
  aQuad.resize(nx*ny*4);
  for(unsigned int iy=0;iy<ny;++iy){
    for(unsigned int ix=0;ix<nx;++ix){
      unsigned int iq = iy*nx+ix;
      aQuad[iq*4+0] = (iy+0)*(nx+1)+(ix+0);
      aQuad[iq*4+1] = (iy+0)*(nx+1)+(ix+1);
      aQuad[iq*4+2] = (iy+1)*(nx+1)+(ix+1);
      aQuad[iq*4+3] = (iy+1)*(nx+1)+(ix+0);
    }
  }
}

// ----------------------------------

// print out error
static void error_callback( int error, const char* description){ fputs(description, stderr); }

int main()
{
  std::vector<float> aXY; // coordinates
  std::vector<unsigned int> aQuad; // index of points
  std::vector<float> aMass; // point mass
  { // initialize mesh and point masses
    const unsigned int nx = 3;
    const unsigned int ny = 10;
    const float scale = 0.1;
    // make grid
    MeshQuad2_Grid(
        aXY, aQuad,
        3, 10);
    // set mass
    aMass.resize(aXY.size() / 2, 1.f);
    for (unsigned int ixy = 0; ixy < aXY.size() / 2; ++ixy){
      if( aXY[ixy*2+1] > 0.001 ){ continue; }
      aMass[ixy] = 10000.0; // mass at the boundary is very large to prevent moving during shape matching operation
    }
    for(unsigned int ixy=0;ixy<aXY.size()/2;++ixy) {
      aXY[ixy*2+0] = (aXY[ixy*2+0] - nx*0.5)*scale;
      aXY[ixy*2+1] = (aXY[ixy*2+1] - ny*0.5)*scale;
    }
  }
  const std::vector<float> aXY0 = aXY; // rest shape
  std::vector<float> aUV(aXY.size(),0.0); // velocity
  std::vector<float> aXYt(aXY.size(),0.0); // tentative shape
  float dt = 0.03;
  // ---------
  delfem2::glfw::CViewer2 viewer;
  {
    viewer.view_height = 1.0;
    viewer.title = "task9: Shape Matching Deformation";
  }
  glfwSetErrorCallback(error_callback);
  if ( !glfwInit() ) { exit(EXIT_FAILURE); }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  // ------
  viewer.InitGL();

  while (!glfwWindowShouldClose(viewer.window))
  {
    { // animation using Shape Matching Deformation [Müller et al. 2005]
      for(unsigned int ixy=0;ixy<aXY.size()/2;++ixy) { // set position & velocity at the boundary
        if( aMass[ixy] < 1.1 ){ continue; }
        aXY[ixy*2+0] = aXY0[ixy*2+0]+0.2*sin(glfwGetTime()*5);
        aXY[ixy*2+1] = aXY0[ixy*2+1];
        aUV[ixy*2+0] = 0.0;
        aUV[ixy*2+1] = 0.0;
      }
      for(unsigned int ixy=0;ixy<aXY.size()/2;++ixy){ // compute tentative shape
        aXYt[ixy*2+0] = aXY[ixy*2+0] + dt*aUV[ixy*2+0];
        aXYt[ixy*2+1] = aXY[ixy*2+1] + dt*aUV[ixy*2+1];
      }
      for(unsigned int iq=0;iq<aQuad.size()/4;++iq){
        const Eigen::Vector2f ap[4] = { // coordinates of quad's corner points (tentative shape)
            Eigen::Map<Eigen::Vector2f>(aXYt.data()+aQuad[iq*4+0]*2),
            Eigen::Map<Eigen::Vector2f>(aXYt.data()+aQuad[iq*4+1]*2),
            Eigen::Map<Eigen::Vector2f>(aXYt.data()+aQuad[iq*4+2]*2),
            Eigen::Map<Eigen::Vector2f>(aXYt.data()+aQuad[iq*4+3]*2) };
        const Eigen::Vector2f aq[4] = { // coordinates of quads' corner points (rest shape)
            Eigen::Map<const Eigen::Vector2f>(aXY0.data()+aQuad[iq*4+0]*2),
            Eigen::Map<const Eigen::Vector2f>(aXY0.data()+aQuad[iq*4+1]*2),
            Eigen::Map<const Eigen::Vector2f>(aXY0.data()+aQuad[iq*4+2]*2),
            Eigen::Map<const Eigen::Vector2f>(aXY0.data()+aQuad[iq*4+3]*2) };
        const float am[4] = { // masses of the points
            aMass[aQuad[iq*4+0]],
            aMass[aQuad[iq*4+1]],
            aMass[aQuad[iq*4+2]],
            aMass[aQuad[iq*4+3]] };
        // write some code below to rigidly transform the points in the rest shape (`aq`) such that the
        // weighted sum of squared distances against the points in the tentative shape (`ap`) is minimized (`am` is the weight).

        //calculate R ???
        Eigen::Matrix2f R = Eigen::Matrix2f::Zero();

        // Calculate sum of mass points
        float m_sum = am[0] + am[1] + am[2] + am[3];

        // Calculate t_opt -> (am*ap)/m_sum - R * (am*aq)/m_sum
        Eigen::Vector2f t_opt[4] = {
            (am[0] * ap[0]) / m_sum - R * ((am[0] * aq[0]) / m_sum) ,
            (am[1] * ap[1]) / m_sum - R * ((am[1] * aq[1]) / m_sum) ,
            (am[2] * ap[2]) / m_sum - R * ((am[2] * aq[2]) / m_sum) ,
            (am[3] * ap[3]) / m_sum - R * ((am[3] * aq[3]) / m_sum) };

        // Calculate R_opt with JacobiSVD
        Eigen::JacobiSVD< Eigen::MatrixXf> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::MatrixXf V = svd.matrixV();
        Eigen::Matrix2f R_opt = U * V.transpose();

        // Update position -> x_i = R_opt * Xi + t_opt;
        // but ap[] is constant and can't be updated.. so maybe update aXYt??
           //ap[0] = R_opt * aq[0] + t_opt[0];
           //ap[1] = R_opt * aq[1] + t_opt[1];
           //ap[2] = R_opt * aq[2] + t_opt[2];
           //ap[3] = R_opt * aq[3] + t_opt[3];
        
        // no edits further down
      }
      for(unsigned int ixy=0;ixy<aXY.size()/2;++ixy) { // update position and velocities
        aUV[ixy*2+0] = (aXYt[ixy*2+0] - aXY[ixy*2+0])/dt;
        aUV[ixy*2+1] = (aXYt[ixy*2+1] - aXY[ixy*2+1])/dt;
        aXY[ixy*2+0] = aXYt[ixy*2+0];
        aXY[ixy*2+1] = aXYt[ixy*2+1];
      }
    }
    // ----------------
    viewer.DrawBegin_oldGL();
    ::glColor3d(0,0,0);
    ::glBegin(GL_LINES);
    for(unsigned int iq=0;iq<aQuad.size()/4;++iq){ // draw edges of quads
      ::glVertex2fv(aXY.data()+aQuad[iq*4+0]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+1]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+1]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+2]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+2]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+3]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+3]*2);
      ::glVertex2fv(aXY.data()+aQuad[iq*4+0]*2);
    }
    ::glEnd();
    viewer.SwapBuffers();
    glfwPollEvents();
  }
  glfwDestroyWindow(viewer.window);
  glfwTerminate();
  exit(EXIT_SUCCESS);
}
