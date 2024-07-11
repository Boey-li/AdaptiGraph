
class by_Granular : public Scene
{
public:

	by_Granular(const char* name) : Scene(name) {}

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

        float radius = ptr[0];

        // add carrots
		int num_x = ptr[1];
		int num_y = ptr[2];
		int num_z = ptr[3];
		float granular_scale = ptr[4];
		float pos_x = ptr[5];
		float pos_y = ptr[6];
		float pos_z = ptr[7];
		float granular_dis = ptr[8];
        int draw_mesh = ptr[9];
		
		float shapeCollisionMargin = ptr[10];
		float collisionDistance = ptr[11];
		float dynamic_friction = ptr[12]; 
		float mass = ptr[13];

		int regular_shape = ptr[14];
		float shape_min_dist = ptr[15]; //6
		float shape_max_dist = ptr[16]; //10


		float inv_mass = 1.0f/mass;
		
		// void CreateParticleShape(const Mesh* srcMesh, 
		// Vec3 lower, Vec3 scale, float rotation, float spacing, 
		// Vec3 velocity, float invMass, bool rigid, float rigidStiffness, 
		// int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, 
		// float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f, bool texture=false)
		int group = 0;
        float pos_diff = granular_scale + granular_dis;
        // add carrots
		for (int x_idx = 0; x_idx < num_x; x_idx++){
			for (int z_idx = 0; z_idx < num_z; z_idx++) {
				// if ((x_idx - (num_x-1)*1.0/2.0) * (x_idx - (num_x-1)*1.0/2.0) + (z_idx - (num_z-1)*1.0/2.0) * (z_idx - (num_z-1)*1.0/2.0) > (num_x*1.0/2.0) * (num_z*1.0/2.0)) {
				// 	continue;
				// }
				for (int y_idx = 0; y_idx < num_y; y_idx++) {
				int num_planes = Rand(6,10); //6-12
				Mesh* m = CreateRandomConvexMesh(num_planes, shape_min_dist, shape_max_dist, regular_shape);
				CreateParticleShape(m, Vec3(pos_x + float(x_idx) * pos_diff, pos_y + float(y_idx) * pos_diff, pos_z + float(z_idx) * pos_diff), 
									granular_scale, 0.0f, radius*1.001f, 
									0.0f, inv_mass, true, 0.8f, 
									NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 
									0.0f, Vec4(237.0f/255.0f, 145.0f/255.0f, 33.0f/255.0f, 1.0f));	
				}
			}
		}

        g_numSubsteps = 12;

		g_params.numIterations = 6;
		g_params.radius = radius;
		g_params.dynamicFriction = dynamic_friction;
		g_params.dissipation = 0.001f;
		g_params.sleepThreshold = g_params.radius*0.2f;
		g_params.relaxationFactor = 1.3f;
		g_params.restitution = 0.0f;
		g_params.shapeCollisionMargin = shapeCollisionMargin; //0.01f
		g_params.collisionDistance = collisionDistance;

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