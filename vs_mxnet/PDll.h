////////////////////////////////////////////////////////////////////////////////////////////////////
//	Class:  PDll																								    //
//  Authors: MicHael Galkovsky                                                                           //
//  Date:    April 14, 1998                                                                                  //
//  Company:  Pervasive Software                                                                    //
//  Purpose:    Base class to wrap dynamic use of dll                                          //
//////////////////////////////////////////////////////////////////////////////////////////////

#if !defined (_PDLL_H_)
#define _PDLL_H_

#include <windows.h>
#include <winbase.h>

#define FUNC_LOADED 3456

// function declarations according to the number of parameters
// define the type
// declare a variable of that type
// declare a member function by the same name as the dll function
// check for dll handle
// if this is the first call to the function then try to load it
// if not then if the function was loaded successfully make a call to it
// otherwise return a NULL cast to the return parameter.

#define DECLARE_FUNCTION0(retVal, FuncName) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName; \
	retVal FuncName() \
	{ \
		if (m_dllHandle) \
		{ \
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(); \
			else \
				return (retVal)NULL; \
		} \
		else \
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION1(retVal, FuncName, Param1) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName(Param1 p1) \
	{ \
		if (m_dllHandle) \
		{ \
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1); \
			else \
				return (retVal)NULL; \
		} \
		else \
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION2(retVal, FuncName, Param1, Param2) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2); \
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION3(retVal, FuncName, Param1, Param2, Param3) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED; \
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3);\
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION4(retVal, FuncName, Param1, Param2, Param3, Param4) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4);\
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION5(retVal, FuncName, Param1, Param2, Param3, Param4, Param5) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName; \
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5);\
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION6(retVal, FuncName, Param1, Param2, Param3, Param4, Param5, Param6) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5, Param6); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5, Param6 p6) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5, p6);\
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION7(retVal, FuncName, Param1, Param2, Param3, Param4, Param5, Param6, Param7) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5, Param6, Param7); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5, Param6 p6, Param7 p7) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5, p6, p7);\
			else \
				return (retVal)NULL; \
		} \
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION8(retVal, FuncName, Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5, Param6 p6, Param7 p7, Param8 p8) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5, p6, p7, p8);\
			else \
				return (retVal)NULL; \
		}\
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION9(retVal, FuncName, Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8, Param9) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8, Param9); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName; \
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5, Param6 p6, Param7 p7, Param8 p8, Param9 p9) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_NAME != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5, p6, p7, p8, p9);\
			else \
				return (retVal)NULL; \
		}\
		else\
			return (retVal)NULL; \
	}

#define DECLARE_FUNCTION10(retVal, FuncName, Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8, Param9, Param10) \
	typedef  retVal (CALLBACK* TYPE_##FuncName)(Param1, Param2, Param3, Param4, Param5, Param6, Param7, Param8, Param9, Param10); \
	TYPE_##FuncName m_##FuncName; \
	short m_is##FuncName;\
	retVal FuncName (Param1 p1, Param2 p2, Param3 p3, Param4 p4, Param5 p5, Param6 p6, Param7 p7, Param8 p8, Param9 p9, Param10 p10) \
	{\
		if (m_dllHandle)\
		{\
			if (FUNC_LOADED != m_is##FuncName) \
			{\
				m_##FuncName = NULL; \
				m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName); \
				m_is##FuncName = FUNC_LOADED;\
			}\
			if (NULL != m_##FuncName) \
				return m_##FuncName(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);\
			else \
				return (retVal)NULL; \
		}\
		else					\
			return (retVal)NULL;\
	}

//declare constructors and LoadFunctions
#define DECLARE_CLASS(ClassName) \
	public:	\
	ClassName (const char* name){LoadDll(name);} \
	ClassName () {PDLL();}

class PDLL
{
protected:
	HINSTANCE m_dllHandle;
	char* m_dllName;
	int m_refCount;

public:

	PDLL()
	{
		m_dllHandle = NULL;
		m_dllName = NULL;
		m_refCount = 0;
	}

	//A NULL here means the name has already been set
	void LoadDll(const char* name, short showMsg = 0)
	{
		if (name)
			SetDllName(name);

		//try to load
		m_dllHandle = LoadLibraryA(m_dllName);

		if (m_dllHandle == NULL && showMsg) {
			//show warning here if needed
			fprintf_s(stderr, "cannot load: %s", m_dllName);
		}
	}

	bool SetDllName(const char* newName)
	{
		bool retVal = false;

		//we allow name resets only if the current DLL handle is invalid
		//once they've hooked into a DLL, the  name cannot be changed
		if (!m_dllHandle) {
			if (m_dllName) {
				delete[]m_dllName;
				m_dllName = NULL;
			}

			//They may be setting this null (e.g., uninitialize)
			if (newName) {
				m_dllName = new char[MAX_PATH + 1];
				//make sure memory was allocated
				if (m_dllName)
					strcpy_s(m_dllName, strlen(newName) + 1, newName);
				else
					retVal = false;
			}
			retVal = true;
		}
		return retVal;
	}

	virtual bool Initialize(short showMsg = 0)
	{
		bool retVal = false;

		//Add one to our internal reference counter
		m_refCount++;

		if (m_refCount == 1 && m_dllName) //if this is first time, load the DLL
		{
			//we are assuming the name is already set
			LoadDll(NULL, showMsg);
			retVal = (m_dllHandle != NULL);
		}
		return retVal;
	}

	virtual void Uninitialize(void)
	{
		//If we're already completely unintialized, early exit
		if (!m_refCount)
			return;

		//if this is the last time this instance has been unitialized,
		//then do a full uninitialization
		m_refCount--;

		if (m_refCount < 1) {
			if (m_dllHandle) {
				FreeLibrary(m_dllHandle);
				m_dllHandle = NULL;
			}

			SetDllName(NULL); //clear out the name & free memory
		}
	}

	virtual bool IsInit()
	{
		return (m_dllHandle != NULL);
	}

	~PDLL()
	{
		//force this to be a true uninitialize
		m_refCount = 1;
		Uninitialize();

		//free name
		if (m_dllName) {
			delete[] m_dllName;
			m_dllName = NULL;
		}
	}
};
#endif
