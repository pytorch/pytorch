# -*- cmake -*-

MACRO(ADD_TORCH_PACKAGE package src luasrc)

  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

  ### C/C++ sources
  IF(src)      

    ADD_LIBRARY(${package} MODULE ${src})
    
    ### Torch packages supposes libraries prefix is "lib"
    SET_TARGET_PROPERTIES(${package} PROPERTIES
      PREFIX "lib"
      IMPORT_PREFIX "lib"
      INSTALL_NAME_DIR "@executable_path/${Torch_INSTALL_BIN2CPATH}")
    
    IF(APPLE)
      SET_TARGET_PROPERTIES(${package} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
    ENDIF()
    
    INSTALL(TARGETS ${package} 
      RUNTIME DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR}
      LIBRARY DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR})
    
  ENDIF(src)
  
  ### lua sources
  IF(luasrc)
    INSTALL(FILES ${luasrc} 
      DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package})
  ENDIF(luasrc)
    
ENDMACRO(ADD_TORCH_PACKAGE)
