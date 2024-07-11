class by_RigidGranular: public Scene
{
public:

	by_RigidGranular(const char* name) : Scene(name) {}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	// void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
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
		int type = ptr[4];
		int draw_mesh = ptr[5];
		
		float radius = ptr[6];
		float mass = ptr[7];
		float rigidStiffness = ptr[8]; //1.0
		float dynamicFriction = ptr[9]; //1.0
		float staticFriction = ptr[10]; //0.0
		float viscosity = ptr[11]; //0.0
		float rotation = ptr[20]; //0.0

		float invMass = 1.0f/mass; //0.25
		int group = 0;
		float s = radius*0.5f;

		char path[100];

		if (type == 1)
			make_path(path, "/data/box.ply");
		else if (type == 3)
			make_path(path, "/data/ycb/03_cracker_box.obj");
		else if (type == 4)
			make_path(path, "/data/ycb/04_sugar_box.obj");
		else if (type == 5)
			make_path(path, "/data/ycb/05_tomato_soup_can.obj");
		else if (type == 6)
			make_path(path, "/data/ycb/06_mustard_bottle.obj");
		else if (type == 7)
			make_path(path, "/data/ycb/07_tuna_fish_can.obj");
		else if (type == 8)
			make_path(path, "/data/ycb/08_pudding_box.obj");
		else if (type == 9)
			make_path(path, "/data/ycb/09_gelatin_box.obj");
		else if (type == 10)
			make_path(path, "/data/ycb/12_strawberry.obj");
		else if (type == 11)
			make_path(path, "/data/ycb/13_apple.obj");
		else if (type == 12)
			make_path(path, "/data/ycb/14_lemon.obj");
		else if (type == 13)
			make_path(path, "/data/ycb/15_peach.obj");	
		else if (type == 14)
			make_path(path, "/data/ycb/16_pear.obj");
		else if (type == 15)
			make_path(path, "/data/ycb/17_orange.obj");
		else if (type == 16)
			make_path(path, "/data/ycb/19_pitcher_base.obj");
		else if (type == 17)
			make_path(path, "/data/ycb/21_bleach_cleanser.obj");
		else if (type == 18)
			make_path(path, "/data/ycb/24_bowl.obj");
		else if (type == 19)
			make_path(path, "/data/ycb/35_power_drill.obj");
		else if (type == 20)
			make_path(path, "/data/ycb/36_wood_block.obj");
		
		// void CreateParticleShape(const Mesh* srcMesh, 
		// Vec3 lower, Vec3 scale, float rotation, float spacing, 
		// Vec3 velocity, float invMass, bool rigid, float rigidStiffness, 
		// int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, 
		// float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f, bool texture=false)
		CreateParticleShape(
		        GetFilePathByPlatform(path).c_str(),
				Vec3(x, y, z),
				scale, rotation, s, Vec3(0.0f, 0.0f, 0.0f), 
				invMass, true, rigidStiffness, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);
			
		// for (int i = 0; i < g_buffers->rigidOffsets.size(); ++i)
		// {
		// 	std::cout << "g_buffers->rigidOffsets" << g_buffers->rigidOffsets[i] << std::endl; 
		// }
		
		// add carrots
		int num_x = ptr[12];
		int num_y = ptr[13];
		int num_z = ptr[14];
		float granular_scale = ptr[15];
		float pos_x = ptr[16];
		float pos_y = ptr[17];
		float pos_z = ptr[18];
		float granular_dis = ptr[19];

		float pos_diff = granular_scale + granular_dis;

		// int num_planes = Rand(6,12);
		// Mesh* m = CreateRandomConvexMesh(num_planes, 5.0f, 10.0f);
		for (int x_idx = 0; x_idx < num_x; x_idx++){
			for (int z_idx = 0; z_idx < num_z; z_idx++) {
				for (int y_idx = 0; y_idx < num_y; y_idx++) {
				int num_planes = Rand(6,12);
				Mesh* m = CreateRandomConvexMesh(num_planes, 5.0f, 10.0f);
				CreateParticleShape(m, Vec3(pos_x + float(x_idx) * pos_diff, pos_y + float(y_idx) * pos_diff, pos_z + float(z_idx) * pos_diff), 
									granular_scale, 0.0f, radius*1.001f,
									0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 
									0.0f, Vec4(237.0f/255.0f, 145.0f/255.0f, 33.0f/255.0f, 1.0f));	
				}
			}
		}

		// for (int i = 0; i < g_buffers->rigidOffsets.size(); ++i)
		// {
		// 	std::cout << "g_buffers->rigidOffsets" << g_buffers->rigidOffsets[i] << std::endl; 
		// }

		g_numSolidParticles = g_buffers->positions.size();
		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.dynamicFriction = dynamicFriction;
		g_params.staticFriction = staticFriction;
		g_params.viscosity = viscosity;

		g_params.numIterations = 12;
		g_params.particleCollisionMargin = g_params.radius*0.25f;	// 5% collision margin
		g_params.sleepThreshold = g_params.radius*0.25f;
		g_params.shockPropagation = 6.f;
		g_params.restitution = 0.2f;
		g_params.relaxationFactor = 1.f;
		g_params.damping = 0.14f;

		float restDistance = radius*0.55f;
		Emitter e1;
		e1.mDir = Vec3(1.0f, 0.0f, 0.0f);
		e1.mRight = Vec3(0.0f, 0.0f, -1.0f);
		e1.mPos = Vec3(radius, 1.f, 0.65f);
		e1.mSpeed = (restDistance/g_dt)*2.0f; // 2 particle layers per-frame
		e1.mEnabled = true;

		g_emitters.push_back(e1);

		g_waveFloorTilt = 0.0f;
		g_waveFrequency = 1.5f;
		g_waveAmplitude = 2.0f;
		
		g_warmup = false;

		if (draw_mesh) {
			g_drawMesh = true;
			g_drawPoints = false;
			g_drawSprings = false;
		} else {
			g_drawMesh = false;
			g_drawPoints = true;
			g_drawSprings = false;
		};
	}
	
};
