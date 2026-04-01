import * as wasmModule from '../slang-wasm.js';
import { readFileSync } from 'fs';
import { resolve, basename } from 'path';

async function runSmokeTest() {
    try {
        // Get the file path from command line arguments
        const filePath = process.argv[2];
        const entryPointName = process.argv[3];
        if (!filePath || !entryPointName) {
            console.error('Please provide a path to a .slang file and an entry point name');
            console.error('Usage: node test/smoke-test.js <path-to-slang-file> <entry-point-name>');
            process.exit(1);
        }

        console.log(`Starting Slang WASM smoke test with file: ${filePath} and entry point: ${entryPointName}`);
        
        // Read the source file
        const absolutePath = resolve(filePath);
        const source = readFileSync(absolutePath, 'utf8');
        const fileName = basename(filePath);
        
        // Load the WASM module
        const module = await wasmModule.default();
        console.log('WASM module loaded successfully');

        // Print available compile targets
        const targets = module.getCompileTargets();
        if (!targets) {
            throw new Error('Failed to get compile targets');
        }
        console.log('Available compile targets:', JSON.stringify(targets, null, 2));

        // Find SPIRV target value
        const spirvTarget = targets.findIndex(target => target.name.toLowerCase() === 'spirv');
        if (spirvTarget === -1) {
            throw new Error('SPIRV target not found in available targets');
        }
        console.log('Found SPIRV target at index:', spirvTarget);

        // Get the actual SPIRV target value
        const spirvTargetValue = targets[spirvTarget].value;
        console.log('SPIRV target value:', spirvTargetValue);

        // Create a global session
        const globalSession = module.createGlobalSession();
        if (!globalSession) {
            throw new Error('Failed to create global session');
        }
        console.log('Global session created');

        // Create a session with SPIRV as the target
        const session = globalSession.createSession(spirvTargetValue);
        if (!session) {
            throw new Error('Failed to create session');
        }
        console.log('Session created with SPIRV target');

        // Load the shader source
        const module1 = session.loadModuleFromSource(source, fileName, '');
        if (!module1) {
            const error = module.getLastError();
            throw new Error(`Failed to load module: ${error ? error.message : 'Unknown error'}`);
        }
        console.log('Shader module loaded');

        // Check for compilation errors
        const error = module.getLastError();
        if (error && error.result !== module.SLANG_OK) {
            throw new Error(`Compilation failed: ${error.message}`);
        }
        console.log('No compilation errors found');

        // Try to find the entry point
        const entryPoint = module1.findEntryPointByName(entryPointName);
        if (!entryPoint) {
            throw new Error(`Could not find entry point "${entryPointName}"`);
        }
        console.log(`Entry point "${entryPointName}" found`);

        // Create and link the program
        const program = session.createCompositeComponentType([module1]);
        if (!program) {
            throw new Error('Failed to create composite component type');
        }
        const linkedProgram = program.link();
        if (!linkedProgram) {
            throw new Error('Failed to link program');
        }
        console.log('Program created and linked successfully');

        // Try to get the SPIRV code
        console.log('\nTrying to generate SPIRV code:');
        const spirvBinary = linkedProgram.getTargetCodeBlob(0); // 0 is the target index
        if (!spirvBinary) {
            throw new Error('Could not generate SPIRV binary');
        }
        console.log('SPIRV binary generated successfully');
        console.log('Generated binary length:', spirvBinary.length);
        
        // Clean up
        linkedProgram.delete();
        program.delete();
        entryPoint.delete();
        module1.delete();
        session.delete();
        globalSession.delete();
        console.log('Smoke test completed successfully');
        process.exit(0); // Explicit success exit code
    } catch (error) {
        console.error('Smoke test failed:', error);
        process.exit(1); // Error exit code
    }
}

runSmokeTest(); 