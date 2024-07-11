class by_MultiYCB : public Scene
{
public:

	by_MultiYCB(const char* name) : Scene(name) {}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	void Initialize(py::array_t<float> scene_params, 
                    py::array_t<float> vertices,
                    py::array_t<int> stretch_edges,
                    py::array_t<int> bend_edges,
                    py::array_t<int> shear_edges,
                    py::array_t<int> faces,
                    int thread_idx = 0)
	{
		auto ptr = (float *) scene_params.request().ptr;

		float x = ptr[0];
		float y = ptr[1];
		float z = ptr[2];
		float scale = ptr[3];

		float radius = 0.1f;
		int group = 0;

		float s = radius*0.5f;
		float m = 0.25f;

		char path_0[100];
		char path_1[100];
		char path_2[100];
		char path_3[100];
		char path_4[100];
		char path_5[100];
		char path_6[100];
		// char path_7[100];
		// char path_8[100];
		char path_9[100];
		char path_10[100];
		char path_11[100];
		char path_12[100];
		// char path_13[100];
		char path_14[100];
		char path_15[100];
		char path_16[100];
		// char path_17[100];
		char path_18[100];
		char path_19[100];
		// char path_20[100];
		
		// char table_path[100];
		// make_path(table_path, "/data/table.obj");
		
		make_path(path_0, "/data/ycb/03_cracker_box.obj");
		make_path(path_1, "/data/ycb/04_sugar_box.obj");
		make_path(path_2, "/data/ycb/05_tomato_soup_can.obj");
		make_path(path_3, "/data/ycb/06_mustard_bottle.obj");
		make_path(path_4, "/data/ycb/07_tuna_fish_can.obj");
		make_path(path_5, "/data/ycb/08_pudding_box.obj");
		make_path(path_6, "/data/ycb/09_gelatin_box.obj");
		// make_path(path_7, "/data/ycb/10_potted_meat_can.obj");
		// make_path(path_8, "/data/ycb/12_strawberry.obj");
		make_path(path_9, "/data/ycb/13_apple.obj");
		make_path(path_10, "/data/ycb/14_lemon.obj");
		make_path(path_11, "/data/ycb/15_peach.obj");	
		make_path(path_12, "/data/ycb/16_pear.obj");
		// make_path(path_13, "/data/ycb/17_orange.obj");
		make_path(path_14, "/data/ycb/19_pitcher_base.obj");
		make_path(path_15, "/data/ycb/21_bleach_cleanser.obj");
		make_path(path_16, "/data/ycb/24_bowl.obj");
		// make_path(path_17, "/data/ycb/25_mug.obj");
		make_path(path_18, "/data/ycb/35_power_drill.obj");
		make_path(path_19, "/data/ycb/36_wood_block.obj");
		// make_path(path_20, "/data/ycb/37_scissors.obj");
		

		
		CreateParticleShape(
				GetFilePathByPlatform(path_1).c_str(),
				Vec3(x, y, z),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_2).c_str(),
				Vec3(x+1, y, z-1.5),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_3).c_str(),
				Vec3(x-1.3, y, z),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_4).c_str(),
				Vec3(x, y, z+1.2),
				scale*0.8, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_0).c_str(),
				Vec3(x-1, y, z),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_18).c_str(),
				Vec3(x-1, y, z+1),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_16).c_str(),
				Vec3(x, 0, z-0.5),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_15).c_str(),
				Vec3(x+1.5, y, z+1),
				scale, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_9).c_str(),
				Vec3(x+0.5, 1., z-0.3),
				scale*0.5, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_12).c_str(),
				Vec3(x-1, y, z-1.5),
				scale*0.5, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		CreateParticleShape(
				GetFilePathByPlatform(path_11).c_str(),
				Vec3(x+1.5, 1., z-0.3),
				scale*0.5, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), 
				m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
		
		

		g_params.radius = radius;
		g_params.fluidRestDistance = radius;
		g_params.numIterations = 4;
		g_params.viscosity = 1.0f;
		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 0.0f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance*0.5f;
		g_params.vorticityConfinement = 120.0f;
		g_params.cohesion = 0.0025f;
		g_params.drag = 0.06f;
		g_params.lift = 0.f;
		g_params.solidPressure = 0.0f;
		g_params.smoothing = 1.0f;
		g_params.relaxationFactor = 1.0f;

		g_numSubsteps = 2;

		g_drawMesh = true;
		g_drawPoints = false;
		g_drawSprings = false;
	}
	
};
