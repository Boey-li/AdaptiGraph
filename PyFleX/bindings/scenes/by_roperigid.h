class by_RopeRigid: public Scene
{

public:
	by_RopeRigid(const char* name) :
		Scene(name),
		mRadius(0.1f),
		mRelaxationFactor(1.0f),
		mPlinth(false),
		plasticDeformation(false)
	{
		const Vec3 colorPicker[7] =
		{
			Vec3(0.0f, 0.5f, 1.0f),
			Vec3(0.797f, 0.354f, 0.000f),
			Vec3(0.000f, 0.349f, 0.173f),
			Vec3(0.875f, 0.782f, 0.051f),
			Vec3(0.01f, 0.170f, 0.453f),
			Vec3(0.673f, 0.111f, 0.000f),
			Vec3(0.612f, 0.194f, 0.394f)
		};
		memcpy(mColorPicker, colorPicker, sizeof(Vec3) * 7);
	}

	float mRadius;
	float mRelaxationFactor;
	bool mPlinth;

	Vec3 mColorPicker[7];

	struct Instance
	{
		Instance(const char* mesh) :

			mFile(mesh),
			mColor(0.5f, 0.5f, 1.0f),

			mScale(2.0f),
			mTranslation(0.0f, 1.0f, 0.0f),
			mRotation(0.0f, 0.0f, 0.0f, 1.0f),

			mClusterSpacing(1.0f),
			mClusterRadius(0.0f),
			mClusterStiffness(0.5f),

			mLinkRadius(0.0f),
			mLinkStiffness(1.0f),

			mGlobalStiffness(0.0f),

			mSurfaceSampling(0.0f),
			mVolumeSampling(4.0f),

			mSkinningFalloff(2.0f),
			mSkinningMaxDistance(100.0f),

			mClusterPlasticThreshold(0.0f),
			mClusterPlasticCreep(0.0f)
		{}

		const char* mFile;
		Vec3 mColor;

		Vec3 mScale;
		Vec3 mTranslation;
		Quat mRotation;

		float mClusterSpacing;
		float mClusterRadius;
		float mClusterStiffness;

		float mLinkRadius;
		float mLinkStiffness;

		float mGlobalStiffness;

		float mSurfaceSampling;
		float mVolumeSampling;

		float mSkinningFalloff;
		float mSkinningMaxDistance;

		float mClusterPlasticThreshold;
		float mClusterPlasticCreep;
	};

	std::vector<Instance> mInstances;

private:

	struct RenderingInstance
	{
		Mesh* mMesh;
		std::vector<int> mSkinningIndices;
		std::vector<float> mSkinningWeights;
		vector<Vec3> mRigidRestPoses;
		Vec3 mColor;
		int mOffset;
	};

	std::vector<RenderingInstance> mRenderingInstances;

	bool plasticDeformation;


public:
	virtual void AddInstance(Instance instance)
	{
		this->mInstances.push_back(instance);
	}

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
		mRadius = radius;

		// rigid parameters
		float type = ptr[1];
		float dimx_rigid = ptr[2];
		float dimy_rigid = ptr[3];
		float dimz_rigid = ptr[4];
		float scale_rigid = ptr[5];
		float mass_rigid = ptr[6];
		float rotation = ptr[7];

		// rope parameters
		Vec3 rope_scale = Vec3(ptr[8], ptr[9], ptr[10]);
		Vec3 rope_trans = Vec3(ptr[11], ptr[12], ptr[13]);

		float clusterSpacing = ptr[14];
		float clusterRadius = ptr[15];
		float clusterStiffness = ptr[16];

		Vec3 rotate_v = Vec3(ptr[17], ptr[18], ptr[19]);
		float rotate_w = ptr[20];
		Quat rope_rotate = Quat(rotate_v, rotate_w);

		float linkRadius = 0.0f;
		float linkStiffness = 1.0f;
		float globalStiffness = 0.0f;
		float surfaceSampling = 0.0f;
		float volumeSampling = 4.0f;
		float skinningFalloff = 5.0f;
		float skinningMaxDistance = 100.0f;
		float clusterPlasticThreshold = 0.0f;
		float clusterPlasticCreep = 0.0f;

		float relaxationFactor = 1.0f;
		mRelaxationFactor = relaxationFactor;
		plasticDeformation = false;

		// others
		float dynamic_friction = ptr[21];
		float static_friction = ptr[22];
		float viscosity = ptr[23];
		float draw_mesh = ptr[24];

		// add rigid bodies
		float rigid_invMass = 1.0f/mass_rigid;
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
		
		
	
		// g_numSolidParticles = g_buffers->positions.size();

		// add rope
		// int ropeStart = g_buffers->positions.size();
		char rope_path[100];
		Instance rope(make_path(rope_path, "/data/rope.obj"));
		rope.mScale = rope_scale;
		rope.mTranslation = rope_trans;
		rope.mRotation = rope_rotate;
		rope.mClusterSpacing = clusterSpacing;
		rope.mClusterRadius = clusterRadius;
		rope.mClusterStiffness = clusterStiffness;
		rope.mLinkRadius = linkRadius;
		rope.mLinkStiffness = linkStiffness;
		rope.mGlobalStiffness = globalStiffness;
		rope.mSurfaceSampling = surfaceSampling;
		rope.mVolumeSampling = volumeSampling;
		rope.mSkinningFalloff = skinningFalloff;
		rope.mSkinningMaxDistance = skinningMaxDistance;
		rope.mClusterPlasticThreshold = clusterPlasticThreshold;
		rope.mClusterPlasticCreep = clusterPlasticCreep;
		AddInstance(rope);
		
		// rigid
		CreateParticleShape(
		        GetFilePathByPlatform(path).c_str(),
				Vec3(dimx_rigid, dimy_rigid, dimz_rigid),
				scale_rigid, rotation, s, Vec3(0.0f, 0.0f, 0.0f), 
				rigid_invMass, true, 1.0, NvFlexMakePhase(group++, 0), true, 0.0f,
				0.0f, 0.0f, Vec4(0.0f), 0.0f, true);

		// build soft bodies
		if (g_buffers->rigidIndices.empty())
			g_buffers->rigidOffsets.push_back(0);
	
		mRenderingInstances.resize(0);

		//std::cout << "group before rope:" << group << std::endl; 
		std::cout << "g_buffers->rigidOffsets:" << g_buffers->rigidOffsets.size() << std::endl; //1
		// number of particles
		std::cout << "g_buffers->positions:" << g_buffers->positions.size() << std::endl; //7021
		std::cout << g_buffers->rigidMeshSize[0] << std::endl; //0
		// rope
		CreateSoftBody(mInstances[0], group++);
		std::cout << "group after rope:" << group << std::endl; 

		// no fluids or sdf based collision
		g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;

		g_numSubsteps = 2;
		g_params.numIterations = 4;

		g_params.radius = radius;
		g_params.dynamicFriction = dynamic_friction;
		g_params.staticFriction = static_friction;
		g_params.viscosity = viscosity;

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

		printf("finish scenes\n");

		

		// expand radius for better self collision
		// g_params.radius *= 1.5f;

		// g_lightDistance *= 1.5f;

		
	}

	void CreateSoftBody(Instance instance, int group = 0, bool texture=false)
	{
		RenderingInstance renderingInstance;

		Mesh* mesh = ImportMesh(GetFilePathByPlatform(instance.mFile).c_str(), texture);
		mesh->Normalize();
		mesh->Transform(ScaleMatrix(instance.mScale*mRadius));
		mesh->Transform(RotationMatrix(instance.mRotation)); 
		mesh->Transform(TranslationMatrix(Point3(instance.mTranslation)));
		// mesh->Transform(TranslationMatrix(Point3(instance.mTranslation))*ScaleMatrix(instance.mScale*mRadius));
		// mesh->Transform(RotationMatrix(instance.mRotation));
		

		renderingInstance.mMesh = mesh;
		renderingInstance.mColor = instance.mColor;
		renderingInstance.mOffset = g_buffers->rigidTranslations.size();

		double createStart = GetSeconds();

		// create soft body definition
		NvFlexExtAsset* asset = NvFlexExtCreateSoftFromMesh(
			(float*)&renderingInstance.mMesh->m_positions[0],
			renderingInstance.mMesh->m_positions.size(),
			(int*)&renderingInstance.mMesh->m_indices[0],
			renderingInstance.mMesh->m_indices.size(),
			mRadius,
			instance.mVolumeSampling,
			instance.mSurfaceSampling,
			instance.mClusterSpacing*mRadius,
			instance.mClusterRadius*mRadius,
			instance.mClusterStiffness,
			instance.mLinkRadius*mRadius,
			instance.mLinkStiffness,
			instance.mGlobalStiffness,
			instance.mClusterPlasticThreshold,
			instance.mClusterPlasticCreep);

		double createEnd = GetSeconds();

		// create skinning
		const int maxWeights = 4;

		renderingInstance.mSkinningIndices.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);
		renderingInstance.mSkinningWeights.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);

		for (int i = 0; i < asset->numShapes; ++i)
			renderingInstance.mRigidRestPoses.push_back(Vec3(&asset->shapeCenters[i * 3]));

		double skinStart = GetSeconds();

		NvFlexExtCreateSoftMeshSkinning(
			(float*)&renderingInstance.mMesh->m_positions[0],
			renderingInstance.mMesh->m_positions.size(),
			asset->shapeCenters,
			asset->numShapes,
			instance.mSkinningFalloff,
			instance.mSkinningMaxDistance,
			&renderingInstance.mSkinningWeights[0],
			&renderingInstance.mSkinningIndices[0]);

		double skinEnd = GetSeconds();

		printf("Created soft in %f ms Skinned in %f\n", (createEnd - createStart)*1000.0f, (skinEnd - skinStart)*1000.0f);

		const int particleOffset = g_buffers->positions.size();
		const int indexOffset = g_buffers->rigidOffsets.back();

		// std::cout << "particleOffset:" << particleOffset << std::endl; //0->7021
		// std::cout << "indexOffset:" << indexOffset << std::endl; //0->7021
		
		std::cout << "asset->numShapeIndices:" << asset->numShapeIndices << std::endl; //3213
		std::cout << "asset->numShapes:" << asset->numShapes << std::endl; //50
		std::cout << "asset->numParticles:" << asset->numParticles << std::endl; //1024

		// add particle data to solver
		for (int i = 0; i < asset->numParticles; ++i)
		{
			g_buffers->positions.push_back(&asset->particles[i * 4]);
			g_buffers->velocities.push_back(0.0f);

			const int phase = NvFlexMakePhase(group, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
			g_buffers->phases.push_back(phase);
		}

		std::cout << "rigidIndices 1:" << g_buffers->rigidIndices.size() << std::endl; //7021
		std::cout << "rigidOffsets 1:" << g_buffers->rigidOffsets.size() << std::endl; //2 -> 0, 7021

		// add shape data to solver
		for (int i = 0; i < asset->numShapeIndices; ++i)
		{
			g_buffers->rigidIndices.push_back(asset->shapeIndices[i] + particleOffset);
		}

		// make cluster
		// g_buffers->rigidOffsets.push_back(indexOffset);
		//asset->numShapes
		//int prevShapeOffset = 0;
		for (int i = 0; i < asset->numShapes; ++i)
		{
			// std::cout << "i:" << i << std::endl;
			// std::cout << "numShapes:" << asset->numShapes << std::endl;
			// std::cout << "rigidOffsets:" << g_buffers->rigidOffsets.size() << std::endl;
			// std::cout << "rigidRotation:" << g_buffers->rigidRotations.size() << std::endl;
			// std::cout << "rigidTranslations:" << g_buffers->rigidTranslations.size() << std::endl;
			// std::cout << "rigidMeshSize:" << g_buffers->rigidMeshSize.size() << std::endl;
			// std::cout << "indexOffset:" << indexOffset << std::endl;
			// std::cout << "asset->shapeOffsets[i]:" << asset->shapeOffsets[i] << std::endl;
			g_buffers->rigidOffsets.push_back(asset->shapeOffsets[i] + indexOffset); //make bug
			//int numMeshParticles = asset->shapeOffsets[i] - prevShapeOffset;
			//prevShapeOffset = asset->shapeOffsets[i];
			//g_buffers->rigidMeshSize.push_back(numMeshParticles);
			// g_buffers->rigidTranslations.push_back(Vec3(&asset->shapeCenters[i * 3])); //make rigid bodies disappear
			g_buffers->rigidRotations.push_back(Quat());
			g_buffers->rigidCoefficients.push_back(asset->shapeCoefficients[i]);
		}

		// add plastic deformation data to solver, if at least one asset has non-zero plastic deformation coefficients, leave the according pointers at NULL otherwise
		if (plasticDeformation)
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
				}
			}
			else
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(0.0f);
					g_buffers->rigidPlasticCreeps.push_back(0.0f);
				}
			}
		}
		else 
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				int oldBufferSize = g_buffers->rigidCoefficients.size() - asset->numShapes;

				g_buffers->rigidPlasticThresholds.resize(oldBufferSize);
				g_buffers->rigidPlasticCreeps.resize(oldBufferSize);

				for (int i = 0; i < oldBufferSize; i++)
				{
					g_buffers->rigidPlasticThresholds[i] = 0.0f;
					g_buffers->rigidPlasticCreeps[i] = 0.0f;
				}

				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
				}

				plasticDeformation = true;
			}
		}

		// add link data to the solver 
		// std::cout << "asset->numSprings:" << asset->numSprings << std::endl; //0
		for (int i = 0; i < asset->numSprings; ++i)
		{
			g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 0]);
			g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 1]);

			g_buffers->springStiffness.push_back(asset->springCoefficients[i]);
			g_buffers->springLengths.push_back(asset->springRestLengths[i]);
		}

		NvFlexExtDestroyAsset(asset);

		mRenderingInstances.push_back(renderingInstance);
	}

	virtual void Draw(int pass)
	{
		if (!g_drawMesh)
			return;

		for (int s = 0; s < int(mRenderingInstances.size()); ++s)
		{
			const RenderingInstance& instance = mRenderingInstances[s];

			Mesh m;
			m.m_positions.resize(instance.mMesh->m_positions.size());
			m.m_normals.resize(instance.mMesh->m_normals.size());
			m.m_indices = instance.mMesh->m_indices;

			for (int i = 0; i < int(instance.mMesh->m_positions.size()); ++i)
			{
				Vec3 softPos;
				Vec3 softNormal;

				for (int w = 0; w < 4; ++w)
				{
					const int cluster = instance.mSkinningIndices[i * 4 + w];
					const float weight = instance.mSkinningWeights[i * 4 + w];

					if (cluster > -1)
					{
						// offset in the global constraint array
						int rigidIndex = cluster + instance.mOffset;

						Vec3 localPos = Vec3(instance.mMesh->m_positions[i]) - instance.mRigidRestPoses[cluster];

						Vec3 skinnedPos = g_buffers->rigidTranslations[rigidIndex] + Rotate(g_buffers->rigidRotations[rigidIndex], localPos);
						Vec3 skinnedNormal = Rotate(g_buffers->rigidRotations[rigidIndex], instance.mMesh->m_normals[i]);

						softPos += skinnedPos*weight;
						softNormal += skinnedNormal*weight;
					}
				}

				m.m_positions[i] = Point3(softPos);
				m.m_normals[i] = softNormal;
			}

			DrawMesh(&m, instance.mColor);
		}
	}

};

