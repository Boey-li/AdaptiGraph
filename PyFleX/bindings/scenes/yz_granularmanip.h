

class yz_GranularManip : public Scene
{
public:

    yz_GranularManip(const char* name) : Scene(name) {}

    char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}


    void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float scale = ptr[0];
        float x = ptr[1];
        float y = ptr[2];
        float z = ptr[3];
        float sFriction = ptr[4];
        float dFriction = ptr[5];
		float radius = ptr[6];

        char path[1000];
        // make_path(path, "/data/sandcastle.obj");
        make_path(path, "/data/box.ply");
        // make_path(path, "/data/bunny.ply");
        
        // void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
        CreateParticleShape(GetFilePathByPlatform(path).c_str(), Vec3(x, y, z), scale, 0.0f, radius*1.0001f, 0.0f, 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide), false, 0.00f);

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
		g_drawPoints = true;		
		g_warmup = false;

		// hack, change the color of phase 0 particles to 'sand'		
		g_colors[0] = Colour(0.805f, 0.702f, 0.401f);	
    }
};
