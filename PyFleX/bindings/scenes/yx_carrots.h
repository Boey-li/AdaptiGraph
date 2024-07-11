class yx_Carrots: public Scene
{
public:

    yx_Carrots(const char* name) : Scene(name) {}

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
        float max_scale = ptr[0];
		float min_scale = ptr[1];
        float x = ptr[2];
        float y = ptr[3];
        float z = ptr[4];
        float sFriction = ptr[5];
        float dFriction = ptr[6];
		float draw_skin_ft = ptr[7];
		float num_carrots_ft = ptr[8];
		float minDist = ptr[9];
		float maxDist = ptr[10];
		float num_x_ft = ptr[11];
		float num_y_ft = ptr[12];
		float num_z_ft = ptr[13];
		float pos_diff = ptr[14];
		float add_singular_ft = ptr[15]; // 0.0f or 1.0f
		float sing_x = ptr[16];
		float sing_y = ptr[17];
		float sing_z = ptr[18];
		float add_noise_ft = ptr[19];
		float radius = ptr[20];
		float mass = ptr[21];

		float inv_mass = 1.0f/mass;
		
		float pos_noise = pos_diff*0.5f;
		int draw_skin = (int) draw_skin_ft;
		int num_carrots = (int) num_carrots_ft;
		int num_x = (int) num_x_ft;
		int num_y = (int) num_y_ft;
		int num_z = (int) num_z_ft;
		bool add_singular = (bool) add_singular_ft;
		bool add_noise = (bool) add_noise_ft;

        // granular pile
		// float radius = 0.075f;

		// float pos_diff = 2.0*scale;
		int group = 0;

		for (int y_idx = 0; y_idx < num_y; y_idx++) {
			if (group > num_carrots) {
				break;
			}
			for (int z_idx = 0; z_idx < num_z; z_idx++) {
				if (group > num_carrots) {
					break;
				}
				for (int x_idx = 0; x_idx < num_x; x_idx++) {
					if (group > num_carrots) {
						break;
					}
					float scale_r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
					float scale = min_scale + (max_scale-min_scale)*scale_r;
					// void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, 
					// Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, 
					// int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
					if (draw_skin) {
						int num_planes = Rand(6,12);
						Mesh* m = CreateRandomConvexMesh(num_planes, minDist, maxDist);
						float x_noise = 0.0;
						float y_noise = 0.0;
						float z_noise = 0.0;
						if (add_noise) {
							x_noise = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * pos_noise - pos_noise;
							y_noise = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * pos_noise - pos_noise;
							z_noise = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * pos_noise - pos_noise;
						}
						CreateParticleShape(m, Vec3(x+float(x_idx)*pos_diff+x_noise, y+float(y_idx)*pos_diff+y_noise, z+float(z_idx)*pos_diff+z_noise), 
											scale, 0.0f, radius*1.001f, 0.0f, inv_mass, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.0f, Vec4(237.0f/255.0f, 145.0f/255.0f, 33.0f/255.0f, 1.0f));
						g_drawPoints = false;
					} else {
						int num_planes = Rand(6,12);
						Mesh* m = CreateRandomConvexMesh(num_planes, minDist, maxDist);
						CreateParticleShape(m, Vec3(x+float(x_idx)*pos_diff, y+float(y_idx)*pos_diff, z+float(z_idx)*pos_diff), 
											scale, 0.0f, radius*1.001f, 0.0f, inv_mass, true, 0.8f, NvFlexMakePhase(group++, 0), false);
						g_drawPoints = true;	
					}
				}
			}
		}
		if (add_singular) {
			float scale_r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			float scale = min_scale + (max_scale-min_scale)*scale_r;
			// void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
			if (draw_skin) {
				int num_planes = Rand(6,12);
				Mesh* m = CreateRandomConvexMesh(num_planes, minDist, maxDist);
				CreateParticleShape(m, Vec3(sing_x, sing_y, sing_z), scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.0f, Vec4(237.0f/255.0f, 145.0f/255.0f, 33.0f/255.0f, 1.0f));
				g_drawPoints = false;
			} else {
				int num_planes = Rand(6,12);
				Mesh* m = CreateRandomConvexMesh(num_planes, minDist, maxDist);
				CreateParticleShape(m, Vec3(sing_x, sing_y, sing_z), scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.0f, Vec4(237.0f/255.0f, 145.0f/255.0f, 33.0f/255.0f, 1.0f));
				g_drawPoints = true;	
			}
		}
        g_numSubsteps = 2;
	
		g_params.radius = radius;
		g_params.staticFriction = sFriction;
		g_params.dynamicFriction = dFriction;
		g_params.viscosity = 0.0f;
		g_params.numIterations = 12;
		g_params.particleCollisionMargin = g_params.radius*0.25f;	// 5% collision margin
		g_params.sleepThreshold = g_params.radius*0.25f;
		g_params.shockPropagation = 6.f;
		g_params.restitution = 0.2f;
		g_params.relaxationFactor = 1.f;
		g_params.damping = 0.14f;
		g_params.numPlanes = 1;
		
		// draw options
		g_warmup = false;

		// hack, change the color of phase 0 particles to 'sand'		
		g_colors[0] = Colour(0.805f, 0.702f, 0.401f);	
    }
};
