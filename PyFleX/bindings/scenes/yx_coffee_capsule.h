

class yx_Coffee_Capsule : public Scene
{
public:

    yx_Coffee_Capsule(const char* name) : Scene(name) {}

    char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}


    void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float cof_scale = ptr[0];
        float cof_x = ptr[1];
        float cof_y = ptr[2];
        float cof_z = ptr[3];
        float sFriction = ptr[4];
        float dFriction = ptr[5];
		float draw_skin_ft = ptr[6];
		float num_coffee_ft = ptr[7];

        float cap_scale = ptr[8];
        float cap_x = ptr[9];
        float cap_y = ptr[10];
        float cap_z = ptr[11];
		float num_capsules_ft = ptr[12];
		// float cap_slices_ft = ptr[13];
		// float cap_segments_ft = ptr[14];

		int draw_skin = (int) draw_skin_ft;
		int num_coffee = (int) num_coffee_ft;
		int num_capsules = (int) num_capsules_ft;
		// int cap_slices = (int) cap_slices_ft;
		// int cap_segments = (int) cap_segments_ft;

        // granular pile
		float radius = 0.075f;

        char cof_path[1000];
        char cap_path[1000];
        // make_path(path, "/data/sandcastle.obj");
        make_path(cof_path, "/data/coffee_bean.ply");
        make_path(cap_path, "/data/capsule.obj");
        // make_path(path, "/data/bunny.ply");
		// Mesh* cap_mesh = CreateCapsule(cap_slices, cap_segments, cap_scale, cap_scale*2.);
        
		int cof_num_x = 6;
		int cof_num_y = 10;
		int cof_num_z = 6;
		float cof_pos_diff = cof_scale;

		int cap_num_x = 6;
		int cap_num_y = 10;
		int cap_num_z = 6;
		float cap_pos_diff = cap_scale;
		int group = 0;

		// draw coffee beans
		for (int y_idx = 0; y_idx < cof_num_y + 1; y_idx++) {
			if (group > num_coffee) {
				break;
			}
			for (int x_idx = 0; x_idx < cof_num_x + 1; x_idx++) {
				if (group > num_coffee) {
					break;
				}
				for (int z_idx = 0; z_idx < cof_num_z + 1; z_idx++) {
					if (group > num_coffee) {
						break;
					}
					// void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
					if (draw_skin) {
						CreateParticleShape(GetFilePathByPlatform(cof_path).c_str(), Vec3(cof_x+float(x_idx)*cof_pos_diff, cof_y+float(y_idx)*cof_pos_diff, cof_z+float(z_idx)*cof_pos_diff), cof_scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.1f, Vec4(0.29f, 0.17f, 0.16f, 1.0f));
						g_drawPoints = false;
					} else {
						CreateParticleShape(GetFilePathByPlatform(cof_path).c_str(), Vec3(cof_x+float(x_idx)*cof_pos_diff, cof_y+float(y_idx)*cof_pos_diff, cof_z+float(z_idx)*cof_pos_diff), cof_scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), false);
						g_drawPoints = true;
					}
				}
			}
		}

		// draw capsules
		for (int y_idx = 0; y_idx < cap_num_y; y_idx++) {
			if (group > num_capsules + num_coffee) {
				break;
			}
			for (int z_idx = 0; z_idx < cap_num_z; z_idx++) {
				if (group > num_capsules + num_coffee) {
					break;
				}
				for (int x_idx = 0; x_idx < cap_num_x; x_idx++) {
					if (group > num_capsules + num_coffee) {
						break;
					}
					// void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
					if (draw_skin) {
						// CreateParticleShape(GetFilePathByPlatform(cap_path).c_str(), Vec3(cap_x+float(x_idx)*cap_pos_diff, cap_y+float(y_idx)*cap_pos_diff, cap_z+float(z_idx)*cap_pos_diff), cap_scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.4f, Vec4(18.0f/255.0f, 138.0f/255.0f, 207.0f/255.0f, 1.0f));
						CreateParticleShape(GetFilePathByPlatform(cap_path).c_str(), Vec3(cap_x+float(x_idx)*cap_pos_diff, cap_y+float(y_idx)*cap_pos_diff, cap_z+float(z_idx)*cap_pos_diff), cap_scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.4f);
					} else {
						CreateParticleShape(GetFilePathByPlatform(cap_path).c_str(), Vec3(cap_x+float(x_idx)*cap_pos_diff, cap_y+float(y_idx)*cap_pos_diff, cap_z+float(z_idx)*cap_pos_diff), cap_scale, 0.0f, radius*1.001f, 0.0f, 0.2f, true, 0.8f, NvFlexMakePhase(group++, 0), false);
					}
				}
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
