// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2017 NVIDIA Corporation. All rights reserved.

#pragma once

// disable some warnings
#if _WIN32
#pragma warning(disable: 4267)  // conversion from 'size_t' to 'int', possible loss of data
#endif

class Scene
{
public:

	Scene(const char* name) : mName(name) {}
	
	// virtual void Initialize(py::array_t<float> scene_params, int thread_idx = 0) = 0;

	virtual void Initialize(
				py::array_t<float> scene_params, 
				py::array_t<float> vertices,
				py::array_t<int> stretch_edges,
				py::array_t<int> bend_edges,
				py::array_t<int> shear_edges,
				py::array_t<int> faces, 
				int thread_idx = 0){};

	virtual void PostInitialize() {}
	
	// update any buffers (all guaranteed to be mapped here)
	virtual void Update(py::array_t<float> update_params) {}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}
	
	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {}
	virtual void CenterCamera() {}

	virtual Matrix44 GetBasis() { return Matrix44::kIdentity; }	

	virtual const char* GetName() { return mName; }

	const char* mName;
};


#include "scenes/yz_bunnybath.h" 
#include "scenes/yz_boxbath.h"
#include "scenes/yz_boxbathext.h"
#include "scenes/yz_dambreak.h"
#include "scenes/yz_rigidfall.h"
#include "scenes/yz_ricefall.h" //5
#include "scenes/yz_softbody.h"
#include "scenes/yz_fluidshake.h"
#include "scenes/yz_fluidiceshake.h"
#include "scenes/yz_massrope.h"
#include "scenes/yz_flag.h" //10
#include "scenes/yz_softrod.h"
#include "scenes/yz_clothrigid.h"
#include "scenes/yz_granular.h"
#include "scenes/yz_bunnygrip.h"
#include "scenes/yz_clothmanip.h" //15
#include "scenes/yz_softfall.h"
#include "scenes/yz_fluidpour.h"
#include "scenes/yz_granularmanip.h"
#include "scenes/yz_fluid_and_box.h"

#include "scenes/yx_coffee.h" //20
#include "scenes/yx_capsule.h" //21
#include "scenes/yx_carrots.h" //22
#include "scenes/yx_coffee_capsule.h" //23

#include "scenes/by_apple.h" //24
#include "scenes/by_singleycb.h" //25
#include "scenes/by_softrope.h" //26
#include "scenes/by_cloth.h" //27
#include "scenes/by_multiycb.h" //28
#include "scenes/by_softgym_cloth.h" //29
#include "scenes/softgym_cloth_2.h" //30

#include "scenes/by_rigidrope.h" //31
#include "scenes/by_rigidgranular.h" //32
#include "scenes/by_rigidcloth.h" //33
#include "scenes/by_ropecloth.h"  //34
#include "scenes/by_granular.h" //35
#include "scenes/by_bowlfluid.h" //36

#include "scenes/by_softbody.h" //37
#include "scenes/by_roperigid.h" //38
#include "scenes/by_ropegranular.h" //39

#include "scenes/adhesion.h"
#include "scenes/armadilloshower.h" 
#include "scenes/bananas.h"
#include "scenes/bouyancy.h"
#include "scenes/bunnybath.h"
#include "scenes/ccdfluid.h"
#include "scenes/clothlayers.h" 
#include "scenes/dambreak.h"
#include "scenes/darts.h"
#include "scenes/debris.h"
#include "scenes/deformables.h"
#include "scenes/fluidblock.h"
#include "scenes/fluidclothcoupling.h"
#include "scenes/forcefield.h"
#include "scenes/frictionmoving.h" 
#include "scenes/frictionramp.h"
#include "scenes/gamemesh.h"
#include "scenes/googun.h"
#include "scenes/granularpile.h"
#include "scenes/granularshape.h" 
#include "scenes/inflatable.h"
#include "scenes/initialoverlap.h"
#include "scenes/lighthouse.h"
#include "scenes/localspacecloth.h"
#include "scenes/localspacefluid.h" 
#include "scenes/lowdimensionalshapes.h"
#include "scenes/melting.h"
#include "scenes/mixedpile.h"
#include "scenes/nonconvex.h"
#include "scenes/parachutingbunnies.h" 
#include "scenes/pasta.h"
#include "scenes/player.h"
#include "scenes/potpourri.h"
#include "scenes/rayleightaylor.h"
#include "scenes/restitution.h" 
#include "scenes/rigidfluidcoupling.h"
#include "scenes/rigidpile.h"
#include "scenes/rigidrotation.h"
#include "scenes/rockpool.h"
#include "scenes/sdfcollision.h"
#include "scenes/shapecollision.h"
#include "scenes/shapechannels.h"
#include "scenes/softbody.h"
#include "scenes/spherecloth.h"
#include "scenes/surfacetension.h"
#include "scenes/tearing.h"
#include "scenes/thinbox.h"
#include "scenes/trianglecollision.h"
#include "scenes/triggervolume.h"
#include "scenes/viscosity.h" 
#include "scenes/waterballoon.h"
