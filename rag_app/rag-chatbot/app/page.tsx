import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import ProblemSection from '@/components/ProblemSection';
import SolutionSection from '@/components/SolutionSection';
import OutputSection from '@/components/OutputSection';
import BenchmarksSection from '@/components/BenchmarksSection';
import QuickStartSection from '@/components/QuickStartSection';
import PipelineSection from '@/components/PipelineSection';
import CtaSection from '@/components/CtaSection';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <ProblemSection />
        <SolutionSection />
        <OutputSection />
        <BenchmarksSection />
        <QuickStartSection />
        <PipelineSection />
        <CtaSection />
      </main>
      <Footer />
    </>
  );
}