class yx_Coffee : public Scene
{
public:

    yx_Coffee(const char* name) : Scene(name) {}

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
        float scale = ptr[0];
        float x = ptr[1];
        float y = ptr[2];
        float z = ptr[3];
        float sFriction = ptr[4];
        float dFriction = ptr[5];
		float draw_skin_ft = ptr[6];
		float radius = ptr[7];

		float num_x_ft = ptr[8];
		float num_y_ft = ptr[9];
		float num_z_ft = ptr[10];
		float pos_diff = ptr[11];
		
		float mass = ptr[12];
		float inv_mass = 1.0f/mass;
		
		int draw_skin = (int) draw_skin_ft;
		int num_x = (int) num_x_ft;
		int num_y = (int) num_y_ft;
		int num_z = (int) num_z_ft;

        // granular pile
		// float radius = 0.075f;

        char path[1000];
		make_path(path, "/data/coffee_bean.ply");
        
		// int num_x = 8;
		// int num_y = 10;
		// int num_z = 8;
		// float pos_diff = scale;
		int group = 0;

		for (int y_idx = 0; y_idx < num_y + 1; y_idx++) {
			for (int x_idx = 0; x_idx < num_x + 1; x_idx++) {
				for (int z_idx = 0; z_idx < num_z + 1; z_idx++) {
					// void CreateParticleShape(const Mesh* srcMesh, 
					// Vec3 lower, Vec3 scale, float rotation, float spacing, 
					// Vec3 velocity, float invMass, bool rigid, float rigidStiffness, 
					// int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, 
					// float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
					if (draw_skin) {
						CreateParticleShape(GetFilePathByPlatform(path).c_str(), 
											Vec3(x+float(x_idx)*pos_diff, y+float(y_idx)*pos_diff, z+float(z_idx)*pos_diff), 
											scale, 0.0f, radius*1.001f, 0.0f, inv_mass, true, 0.8f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.1f, Vec4(0.29f, 0.17f, 0.16f, 1.0f));
						g_drawPoints = false;
					} else {
						CreateParticleShape(GetFilePathByPlatform(path).c_str(), Vec3(x+float(x_idx)*pos_diff, y+float(y_idx)*pos_diff, z+float(z_idx)*pos_diff), 
						scale, 0.0f, radius*1.001f, 0.0f, inv_mass, true, 0.8f, NvFlexMakePhase(group++, 0), false);
						g_drawPoints = true;
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
