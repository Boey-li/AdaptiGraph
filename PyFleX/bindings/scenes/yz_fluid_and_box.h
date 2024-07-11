
class yz_FluidAndBox : public Scene
{
public:

	yz_FluidAndBox(const char* name) : Scene(name) {}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	float rand_float(float LO, float HI) {
        return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(HI-LO)));
    }

    void swap(float* a, float* b) {
	    float tmp = *a;
	    *a = *b;
	    *b = tmp;
	}

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{
	    // scene_params: shape [15]
	    // px_f, py_f, pz_f, sx_f, sy_f, sz_f, vis_f, (xyz position, size and viscosity of the fluid block)
	    // px_r, py_r, pz_r, sx_r, sy_r, sz_r, (xyz position and size of the rigid body)
	    // box_dis_x, box_dis_y, (size of the fully-actuated container)

        float radius = 0.1f;
		float restDistance = radius*0.55f;


	    auto ptr = (float *) scene_params.request().ptr;

        int n_fluid = (int) ptr[0];

        for (int i = 0; i < n_fluid; i++) {
	        float px_f = ptr[i * 7 + 1];
	        float py_f = ptr[i * 7 + 2];
	        float pz_f = ptr[i * 7 + 3];
	        float sx_f = ptr[i * 7 + 4];
	        float sy_f = ptr[i * 7 + 5];
	        float sz_f = ptr[i * 7 + 6];
	        float invMass = ptr[i * 7 + 7];
		    
		    // Initialize the fluid block
		    // void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
		    CreateParticleGrid(
		        Vec3(px_f, py_f, pz_f), sx_f, sy_f, sz_f, restDistance, Vec3(0.0f),
                invMass, false, 0.0f,
                NvFlexMakePhase(0, eNvFlexPhaseSelfCollide|eNvFlexPhaseFluid),
                0.005f);
        }


        int st_idx = n_fluid * 7 + 1;
        int n_rigid = (int) ptr[st_idx];

		char box_path[100];
		make_path(box_path, "/data/box.ply");

		int group = 1;
		float s = radius * 0.5f;

        for (int i = 0; i < n_rigid; i++) {
		    // Initialize the rigid block
		    float px_r = ptr[st_idx + i * 10 + 1];    // position x
		    float py_r = ptr[st_idx + i * 10 + 2];	// position y
		    float pz_r = ptr[st_idx + i * 10 + 3];	// position z
		    float sx_r = ptr[st_idx + i * 10 + 4];    // size x
		    float sy_r = ptr[st_idx + i * 10 + 5];   // size y
		    float sz_r = ptr[st_idx + i * 10 + 6];   // size z
            float invMass = ptr[st_idx + i * 10 + 7];
            float R = ptr[st_idx + i * 10 + 8];
            float G = ptr[st_idx + i * 10 + 9];
            float B = ptr[st_idx + i * 10 + 10];

            Vec4 color = Vec4(R, G, B, 0.0f);
            
            // void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
            CreateParticleShape(
                GetFilePathByPlatform(box_path).c_str(), Vec3(px_r, py_r, pz_r),
                Vec3(sx_r, sy_r, sz_r), 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), invMass, true, 1.0f,
                NvFlexMakePhase(group++, 0), true, 0.0f, 0.0f, 0.0f, color);
        }


        float vis_f = ptr[n_fluid * 7 + n_rigid * 10 + 2];
		float draw_mesh = ptr[n_fluid * 7 + n_rigid * 10 + 3];

		g_lightDistance *= 0.5f;

		g_sceneLower = Vec3(-2.0f, 0.0f, -1.0f);
		g_sceneUpper = Vec3(2.0f, 1.0f, 1.0f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.dynamicFriction = 0.01f; 
		g_params.viscosity = vis_f;
		g_params.numIterations = 4;
		g_params.vorticityConfinement = 40.0f;
		g_params.fluidRestDistance = restDistance;
		g_params.solidPressure = 0.f;
		g_params.relaxationFactor = 0.0f;
		g_params.cohesion = 0.02f;
		g_params.collisionDistance = 0.01f;		

		g_maxDiffuseParticles = 0;
		g_diffuseScale = 0.5f;

		g_fluidColor = Vec4(0.113f, 0.425f, 0.55f, 1.f);

		Emitter e1;
		e1.mDir = Vec3(1.0f, 0.0f, 0.0f);
		e1.mRight = Vec3(0.0f, 0.0f, -1.0f);
		e1.mPos = Vec3(radius, 1.f, 0.65f);
		e1.mSpeed = (restDistance/g_dt)*2.0f; // 2 particle layers per-frame
		e1.mEnabled = true;

		g_emitters.push_back(e1);

		// g_numExtraParticles = 48*1024;

		g_lightDistance = 1.8f;

		// g_params.numPlanes = 5;

		g_waveFloorTilt = 0.0f;
		g_waveFrequency = 1.5f;
		g_waveAmplitude = 2.0f;
		
		g_warmup = false;

		// draw options
		if (!draw_mesh) {
			g_drawPoints = true;
			g_drawMesh = false;
			g_drawEllipsoids = false;
		} else {
			g_drawPoints = false;
			g_drawEllipsoids = true;
		}

		g_drawDiffuse = true;
	}

	bool mDam;
};
