def GEN(N):
    results = ''
    def DEF(x): 
        nonlocal results
        results += x

    DEF(f'#define DECLARE_FUNCTION{N}(retVal, FuncName, ')

    for i in range(1, N):
        DEF(f'Param{i}, ')
    DEF(f'Param{N})\\\n')

    DEF('    typedef retVal (__stdcall* TYPE_##FuncName)(')

    for i in range(1, N):
        DEF(f'Param{i}, ')
    DEF(f'Param{N}); \\')

    DEF("""
        TYPE_##FuncName m_##FuncName; \\
        short m_is##FuncName;\\
        retVal FuncName (""")

    for i in range(1, N):
        DEF(f'Param{i} p{i}, ')
    DEF(f'Param{N} p{N})\\')

    DEF("""
        {\\
            if (m_dllHandle) {\\
                if (FUNC_LOADED != m_is##FuncName) {\\
                    m_##FuncName = nullptr;\\
                    m_##FuncName = (TYPE_##FuncName)GetProcAddress(m_dllHandle, #FuncName);\\
                    m_is##FuncName = FUNC_LOADED;\\
                }\\
                if (nullptr != m_##FuncName) return m_##FuncName(""")

    for i in range(1, N):
        DEF(f'p{i}, ')
    DEF(f'p{N});\\')

    DEF("""
                else return (retVal)0;\\
            }\\
            else return (retVal)0;\\
        }
    """)

    print(results)

for i in range(1, 16):
    GEN(i)
